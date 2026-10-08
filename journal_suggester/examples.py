"""Identical candidate-grouped evidence for Kev training and inference."""
import random

from .io import digest

INSTRUCTION = "Which candidate journal is the most plausible publication venue for this manuscript, based on its mathematical content and the reference papers?"


def build_request(query, candidates, names, tokenizer=None, label=None, seed=42,
                  query_tokens=768, reference_tokens=100, max_state=3456, max_request=4096,
                  candidate_order="shuffle"):
    if not candidates:
        raise ValueError("No reference journals available")
    order = list(candidates)
    if candidate_order == "retrieval":
        order.sort(key=lambda c: (-c["score"], c["journal_id"]))
    elif candidate_order == "shuffle":
        random.Random(f"{seed}:{query.get('paper_id', digest([query['title'], query['abstract']]))}").shuffle(order)
    else:
        raise ValueError("Unknown candidate ordering")
    criteria = {c["journal_id"]: names[c["journal_id"]] for c in order}
    if label is not None and label not in criteria:
        raise ValueError("Training label is absent from candidates")

    def truncate(text, limit):
        if tokenizer is None:
            return " ".join(text.split()[:limit])
        ids = tokenizer.encode(text, add_special_tokens=False)
        return tokenizer.decode(ids[:limit], skip_special_tokens=True)

    def render(limit):
        parts = ["Manuscript:\n" + truncate(query["title"] + "\n" + query["abstract"], query_tokens)]
        for candidate in order:
            parts.append("Candidate journal: " + names[candidate["journal_id"]])
            for paper in candidate["references"]:
                parts.append("Reference: " + truncate(paper["title"] + "\n" + paper["abstract"], limit))
            for _ in range(2 - len(candidate["references"])):
                parts.append("Reference: No eligible reference available.")
        question = {"type": "choice", "instructions": INSTRUCTION, "criteria": criteria}
        if label is not None:
            question["label"] = label
        return {"state": "\n\n".join(parts), "questions": {"journal": question}}

    request = render(reference_tokens)
    lengths = {"tokenizer": "word-smoke-only" if tokenizer is None else getattr(tokenizer, "name_or_path", "provided")}
    if tokenizer is not None:
        # Use the actual upstream rendering and strict encoder, not JSON length estimates.
        from kev.api import SystemOneRequest, to_record
        from kev.model import encode, training_context
        for limit in range(reference_tokens, -1, -5):
            request = render(limit)
            body = {"state": request["state"], "questions": {"journal": {k: v for k, v in request["questions"]["journal"].items() if k != "label"}}}
            record, _ = to_record(SystemOneRequest.model_validate(body))
            try:
                context = training_context(max_state)
                enc = encode(tokenizer, record, max_state=max_state, max_branch=context["max_branch"], strict=True)
                if len(enc["ids"]) <= max_request:
                    lengths.update(state_tokens=enc["state_tokens"], request_tokens=len(enc["ids"]), reference_limit=limit)
                    break
            except ValueError:
                pass
        else:
            raise ValueError("Query and question exceed token budget even without evidence")
    return request, lengths

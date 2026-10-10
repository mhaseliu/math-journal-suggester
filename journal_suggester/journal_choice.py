"""The evaluated title/abstract and alphabetical 95-journal request format."""

INSTRUCTION = "Which candidate journal is the most plausible publication venue for this manuscript, based on its mathematical content?"


def ordered_journals(rows):
    """Use one target-independent order, with all 95 distinct journal IDs."""
    if isinstance(rows, dict):
        rows = [{"journal_id": key, "journal_name": value} for key, value in rows.items()]
    result = [{"journal_id": row["journal_id"], "journal_name": row["journal_name"]} for row in rows]
    if len(result) != 95 or len({r["journal_id"] for r in result}) != 95:
        raise ValueError("Expected exactly 95 unique journals")
    if any(not isinstance(r[k], str) or not r[k].strip() for r in result for k in r):
        raise ValueError("Journal IDs and names must be nonempty strings")
    if len({r["journal_name"].casefold() for r in result}) != 95:
        raise ValueError("Journal display names must be unique")
    return sorted(result, key=lambda r: (r["journal_name"].casefold(), r["journal_id"]))


def _strict_encode(body, tokenizer, config, label=None):
    from kev.api import SystemOneRequest, to_record
    from kev.model import encode, training_context
    record, info = to_record(SystemOneRequest.model_validate(body))
    criteria = body["questions"]["journal"]["criteria"]
    keys = list(criteria)
    if (len(info) != 1 or info[0]["keys"] != keys or len(record["questions"]) != 1
            or record["questions"][0]["options"] != keys or any(value is not None for value in criteria.values())):
        raise ValueError("Kev changed journal IDs, names or option ordering")
    label_index = keys.index(label) if label is not None else 0
    record["questions"][0]["label"] = label_index
    context = training_context(config["max_state"])
    encoded = encode(tokenizer, record, max_state=config["max_state"],
                     max_branch=context["max_branch"], strict=True)
    if (encoded["state_truncated"] or len(encoded["opt_idx"]) != 1
            or len(encoded["opt_idx"][0]) != 95 or encoded["labels"] != [label_index]
            or {i for i in encoded["opt"] if i >= 0} != set(range(95))):
        raise ValueError("Kev dropped choices, truncated input or changed label mapping")
    return encoded


def build_request(query, journals, tokenizer, *, label=None, config):
    """Render only title/abstract and all journal names, then strictly encode it."""
    if tokenizer is None:
        raise ValueError("Supply the pinned Kev tokenizer for strict preparation")
    catalog = ordered_journals(journals)
    names = {row["journal_id"]: row["journal_name"] for row in catalog}
    criteria = {row["journal_name"]: None for row in catalog}
    if label is not None and label not in names:
        raise ValueError("Training label is absent from journal catalog")
    ids = tokenizer.encode(query["title"] + "\n" + query["abstract"], add_special_tokens=False)
    state = "Manuscript:\n" + tokenizer.decode(ids[:config["query_tokens"]], skip_special_tokens=True)
    question = {"type": "choice", "instructions": INSTRUCTION, "criteria": criteria}
    body = {"state": state, "questions": {"journal": question}}
    choice_label = names[label] if label is not None else None
    encoded = _strict_encode(body, tokenizer, config, choice_label)
    if len(encoded["ids"]) > config["max_request"] or encoded["state_tokens"] > config["max_state"]:
        raise ValueError("Request exceeds context budget with all 95 journal choices")
    # Validation targets remain outside the body passed to the encoder/model.
    if label is not None:
        question["label"] = choice_label
    lengths = {"tokenizer": getattr(tokenizer, "name_or_path", "provided"),
               "state_tokens": encoded["state_tokens"], "request_tokens": len(encoded["ids"]),
               "query_tokens": min(len(ids), config["query_tokens"]), "choice_count": len(criteria)}
    return body, lengths

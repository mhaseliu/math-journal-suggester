"""Native pinned Kev trainer with its unconditional option permutation disabled."""
import importlib.metadata
import json
import random


def verify_kev_revision(expected):
    provenance = json.loads(importlib.metadata.distribution("kev").read_text("direct_url.json"))
    if provenance.get("vcs_info", {}).get("commit_id") != expected:
        raise RuntimeError("Fixed-order training requires the pinned Kev code revision")


def ordered_variants(request, args, epoch, pairs=None):
    from kev.data import source_seed
    if pairs is not None or any(getattr(args, key) != 0 for key in
                               ("p_none", "p_none_distract", "p_distract", "p_none_pair", "perm_kl")):
        raise ValueError("Fixed-order training requires all option augmentations disabled")
    rng = random.Random(source_seed(args.seed, f"{epoch}:{request['_meta']['id']}"))
    return [request], rng

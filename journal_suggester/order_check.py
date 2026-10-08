"""Content-preserving candidate permutations and probability comparisons."""
from copy import deepcopy
import math


def reverse_request(request, *, evidence=True):
    result = deepcopy(request)
    criteria = result["questions"]["journal"]["criteria"]
    result["questions"]["journal"]["criteria"] = dict(reversed(list(criteria.items())))
    if evidence:
        separator = "\n\nCandidate journal: "
        query, *blocks = result["state"].split(separator)
        if [b.split("\n", 1)[0] for b in blocks] != list(criteria.values()):
            raise ValueError("Evidence order does not match the choice list")
        result["state"] = query + separator + separator.join(reversed(blocks))
    return result


def validate_probabilities(probabilities, candidates):
    if set(probabilities) != set(candidates):
        raise ValueError("Probability keys do not match candidates")
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("Non-finite or out-of-range probability")
    if abs(sum(probabilities.values()) - 1) > .02:
        raise ValueError("Probabilities are not normalized")


def averaged_probabilities(left, right):
    validate_probabilities(left, right)
    validate_probabilities(right, left)
    return {k: (left[k] + right[k]) / 2 for k in left}

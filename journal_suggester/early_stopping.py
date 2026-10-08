"""Predeclared validation stopping rule; independent of model/framework execution."""
import math


def conditional_cross_entropy(rows):
    included = [r for r in rows if r["target"] in r["probabilities"]]
    if not included:
        raise ValueError("No validation targets have a candidate probability")
    probabilities = [r["probabilities"][r["target"]] for r in included]
    if not all(math.isfinite(p) and 0 < p <= 1 for p in probabilities):
        raise ValueError("Nonpositive/nonfinite target probability")
    return {"n": len(included), "target_absent": len(rows)-len(included),
            "cross_entropy": math.fsum(-math.log(p) for p in probabilities)/len(included)}


class EarlyStopping:
    def __init__(self, policy):
        self.policy = dict(policy)
        self.loss_anchor = None
        self.best_top3 = None
        self.bad_checks = 0
        self.last_epoch = 0

    def update(self, epoch, loss, top3):
        if epoch <= self.last_epoch or not math.isfinite(loss) or not 0 <= top3 <= 1:
            raise ValueError("Invalid or out-of-order validation measurement")
        loss_improved = self.loss_anchor is None or self.loss_anchor-loss >= self.policy["loss_min_delta"]
        top3_improved = self.best_top3 is None or top3 > self.best_top3
        if loss_improved:
            self.loss_anchor = loss
        if top3_improved:
            self.best_top3 = top3
        self.bad_checks = 0 if loss_improved or top3_improved else self.bad_checks+1
        self.last_epoch = epoch
        return {"stop": epoch >= self.policy["minimum_epochs"] and self.bad_checks >= self.policy["patience"],
                "bad_checks": self.bad_checks, "loss_improved": loss_improved, "top3_improved": top3_improved,
                "loss_anchor": self.loss_anchor, "best_top3": self.best_top3}

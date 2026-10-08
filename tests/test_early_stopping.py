import math
import unittest

from journal_suggester.early_stopping import EarlyStopping, conditional_cross_entropy
from journal_suggester.b300_full_train import patched_main


class EarlyStoppingTests(unittest.TestCase):
    def policy(self):
        return EarlyStopping({"minimum_epochs": 2, "patience": 3, "loss_min_delta": .01})

    def test_stall_requires_patience_and_two_epochs(self):
        p = self.policy()
        self.assertFalse(p.update(.5, 2.2, .5)["stop"])
        self.assertFalse(p.update(1, 2.201, .5)["stop"])
        self.assertFalse(p.update(1.5, 2.202, .499)["stop"])
        self.assertTrue(p.update(2, 2.203, .499)["stop"])

    def test_top3_improvement_prevents_loss_only_stop(self):
        p = self.policy()
        for ep in [.5, 1, 1.5]:
            p.update(ep, 2.2, .5)
        result = p.update(2, 2.3, .501)
        self.assertFalse(result["stop"])
        self.assertEqual(result["bad_checks"], 0)

    def test_small_loss_improvements_can_accumulate(self):
        p = self.policy()
        p.update(.5, 2.2, .5)
        self.assertFalse(p.update(1, 2.194, .5)["loss_improved"])
        self.assertTrue(p.update(1.5, 2.188, .5)["loss_improved"])

    def test_loss_conditions_on_available_target_without_probability_clipping(self):
        rows = [{"target": "a", "probabilities": {"a": .25}}, {"target": "b", "probabilities": {"a": 1.0}}]
        r = conditional_cross_entropy(rows)
        self.assertEqual((r["n"], r["target_absent"]), (1, 1))
        self.assertAlmostEqual(r["cross_entropy"], math.log(4))
        rows[0]["probabilities"]["a"] = 0
        with self.assertRaises(ValueError):
            conditional_cross_entropy(rows)

    def test_hook_stops_both_loops_and_preserves_schedule(self):
        source = '''def main():
    steps = 6
    out = []
    if True:
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    for ep in range(3):
        for step in range(ep*2+1, ep*2+3):
            if True:
                out.append(step)
                if step == steps: break
        if step == steps: break
    return out
'''
        class Clock:
            @staticmethod
            def time(): return 0
        namespace = {"journal_apply_precision": lambda: None, "time": Clock,
                     "journal_step_hook": lambda v: True if v["step"] == 4 else None}
        exec(patched_main(source), namespace)
        self.assertEqual(namespace["main"](), [1, 2, 3, 4])
        with self.assertRaises(RuntimeError):
            patched_main(source.replace("if step == steps: break", "pass"))


if __name__ == "__main__":
    unittest.main()

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from journal_suggester.io import digest, request_digest, write_json, write_jsonl
from journal_suggester.training.train import (
    Top3Stopping, check_trial_policy, checked_encoding, file_hash, load_inputs, patched_main, plan,
    require_completed, run, select_checkpoint, select_rate, validate_rows,
)


class KevOnlyTrainingTests(unittest.TestCase):
    def spec(self):
        return json.loads((Path(__file__).resolve().parents[1] / "configs/kev-only.json").read_text())

    def rows(self, training=False, n=1, prefix="paper"):
        candidates = [f"journal-{i:02d}" for i in range(95)]
        names = [f"Journal {i:02d}" for i in range(95)]
        requests, metadata = [], []
        for i in range(n):
            request = {"state": f"Manuscript:\nA theorem {prefix}{i}",
                       "questions": {"journal": {"type": "choice", "instructions": "Choose a journal.",
                                      "criteria": dict.fromkeys(names)}}}
            if training:
                request["questions"]["journal"]["label"] = names[i % 95]
            requests.append(request)
            metadata.append({"paper_id": f"{prefix}:{i}", "group_id": f"group:{prefix}:{i}",
                             "target": candidates[i % 95], "candidates": candidates,
                             "choice_keys": names, "request_hash": request_digest(request)})
        return requests, metadata, candidates, names

    def test_name_only_labels_and_canonical_ids_remain_separate(self):
        requests, metadata, candidates, names = self.rows(training=True)
        validate_rows(requests, metadata, candidates, True, names)
        requests[0]["questions"]["journal"]["label"] = candidates[0]
        metadata[0]["request_hash"] = request_digest(requests[0])
        with self.assertRaisesRegex(ValueError, "training label"):
            validate_rows(requests, metadata, candidates, True, names)

    def test_labels_reordering_dropped_choices_and_descriptions_are_rejected(self):
        for mutation in ("label", "reorder", "drop", "description", "metadata"):
            requests, metadata, candidates, names = self.rows()
            question = requests[0]["questions"]["journal"]
            if mutation == "label":
                question["label"] = names[0]
            elif mutation == "reorder":
                question["criteria"] = dict(reversed(list(question["criteria"].items())))
            elif mutation == "drop":
                del question["criteria"][names[-1]]
            elif mutation == "description":
                question["criteria"][names[0]] = "A journal profile"
            else:
                metadata[0]["choice_keys"] = list(reversed(names))
            metadata[0]["request_hash"] = request_digest(requests[0])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_rows(requests, metadata, candidates, False, names)

    def test_actual_schedules_and_three_update_pilot(self):
        spec = self.spec()
        expected = {"pilot": (3, 3, 8), "screen": (250, 125, 1000), "full": (5000, 500, 1000)}
        for mode, (steps, interval, validation) in expected.items():
            schedule = plan(spec, mode)
            self.assertEqual((schedule["optimizer_steps"], schedule["validation_every_steps"], schedule["validation_papers"]),
                             (steps, interval, validation))
        spec["screen"]["training_papers"] = 1001
        with self.assertRaises(ValueError):
            plan(spec, "screen")

    def test_top3_only_stopping_and_earlier_checkpoint_ties(self):
        policy = Top3Stopping(minimum_epochs=2, patience=3)
        self.assertFalse(policy.update(.5, .7)["stop"])
        self.assertFalse(policy.update(1, .7)["stop"])
        self.assertFalse(policy.update(1.5, .69)["stop"])
        self.assertTrue(policy.update(2, .7)["stop"])
        self.assertEqual(policy.update(2.5, .701)["bad_checks"], 0)
        with self.assertRaises(ValueError):
            policy.update(2.5, .8)
        records = [{"epoch": 2, "metrics": {"top3": .7}}, {"epoch": 1, "metrics": {"top3": .7}},
                   {"epoch": 3, "metrics": {"top3": .69}}]
        self.assertEqual(select_checkpoint(records)["epoch"], 1)

    def test_rate_selection_requires_all_three_and_breaks_ties_by_rate(self):
        rates = [1e-5, 2e-5, 4e-5]
        trials = [{"learning_rate": rate, "mode": "screen", "checks_pass": True,
                   "selected": {"epoch": 2, "metrics": {"top3": .6}}} for rate in rates]
        self.assertEqual(select_rate(trials, rates)["learning_rate"], 1e-5)
        trials[-1]["selected"]["metrics"]["top3"] = .61
        self.assertEqual(select_rate(trials, rates)["learning_rate"], 4e-5)
        for bad in (trials[:2], [trials[0]] * 3):
            with self.assertRaises(ValueError):
                select_rate(bad, rates)

    def test_encoded_options_and_context_must_survive(self):
        candidates = list(range(95))
        encoded = {"ids": [1, 2], "opt_idx": [list(range(95))], "state_truncated": False}
        checked_encoding(encoded, candidates, 6144)
        for changed in ({**encoded, "opt_idx": [list(range(94))]}, {**encoded, "state_truncated": True},
                        {**encoded, "ids": list(range(6145))}):
            with self.assertRaises(ValueError):
                checked_encoding(changed, candidates, 6144)

    def test_pinned_hook_keeps_native_optimization_and_can_stop_both_loops(self):
        source = '''def main():
    steps, step = 6, 0
    if True:
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    for ep in range(3):
        for mb in range(2):
            if True:
                if not a.full_ft: norm = float(torch.nn.utils.clip_grad_norm_((), 1))
                step += 1
                if step == steps: break
        if step == steps: break
    return step
'''
        modified = patched_main(source)
        from types import SimpleNamespace
        calls = []
        namespace = {"torch": SimpleNamespace(nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda *_: 1))),
                     "a": SimpleNamespace(full_ft=False), "time": SimpleNamespace(time=lambda: 0),
                     "journal_apply_precision": lambda: None,
                     "journal_gradient_hook": lambda values: calls.append(values["step"]),
                     "journal_step_hook": lambda values: True if values["step"] == 4 else None}
        exec(modified, namespace)
        self.assertEqual(namespace["main"](), 4)
        self.assertEqual(calls, [0, 1, 2, 3])
        with self.assertRaises(RuntimeError):
            patched_main(source.replace("if step == steps: break", "pass"))

    def fixture(self, root):
        spec = self.spec()
        spec["validation_papers"] = 8
        for mode in ("screen", "full"):
            spec[mode]["training_papers"] = 32
        _, _, candidates, names = self.rows()
        selection = {"config": spec, "models": {}, "candidate_ids": candidates, "choice_keys": names}
        write_json(root / "selection.json", selection)
        files = {}
        for mode in ("pilot", "screen", "full"):
            n = spec[mode]["training_papers"]
            for partition in ("train", "validation"):
                requests, metadata, _, _ = self.rows(partition == "train", n if partition == "train" else 8, partition)
                for suffix, rows in (("", requests), (".meta", metadata)):
                    relative = f"inputs/{mode}/{partition}{suffix}.jsonl"
                    write_jsonl(root / relative, rows)
                    files[relative] = {"bytes": (root / relative).stat().st_size, "sha256": file_hash(root / relative)}
        prepared = {"checks_pass": True, **selection, "selection_hash": digest(selection), "files": files}
        write_json(root / "prepared.json", prepared)
        return selection, prepared

    def test_file_tampering_or_settings_drift_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection, prepared = self.fixture(root)
            with patch("journal_suggester.training.data.load_selection", return_value=(selection, {})):
                load_inputs(root, "pilot")
                altered = copy.deepcopy(prepared)
                altered["config"]["seed"] += 1
                write_json(root / "prepared.json", altered)
                with self.assertRaisesRegex(ValueError, "settings"):
                    load_inputs(root, "pilot")
                write_json(root / "prepared.json", prepared)
                with (root / "inputs/full/train.jsonl").open("a") as stream:
                    stream.write("\n")
                with self.assertRaisesRegex(ValueError, "input changed"):
                    load_inputs(root, "pilot")

    def test_failed_or_wrong_pilot_cannot_unlock_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pilot-result.json"
            good = {"status": "complete", "checks_pass": True, "prepared_hash": "expected", "mode": "pilot"}
            write_json(path, good)
            self.assertEqual(require_completed(path, "expected", "pilot"), good)
            for changed in ({**good, "checks_pass": False}, {**good, "status": "trained"}, {**good, "prepared_hash": "old"}):
                write_json(path, changed)
                with self.assertRaises(ValueError):
                    require_completed(path, "expected", "pilot")

    def test_existing_output_is_never_overwritten_and_laptop_cannot_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection, prepared = self.fixture(root)
            output = root / "trials/pilot"
            output.mkdir(parents=True)
            (output / "keep").write_text("important")
            with patch("journal_suggester.training.data.load_selection", return_value=(selection, {})), patch("platform.machine", return_value="aarch64"):
                with self.assertRaises(FileExistsError):
                    run(root, "pilot")
            self.assertEqual((output / "keep").read_text(), "important")
            with patch("platform.machine", return_value="x86_64"), self.assertRaisesRegex(RuntimeError, "GB10-only"):
                run(root, "pilot")

    def test_direct_full_worker_still_requires_three_completed_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = {"config": self.spec()}
            prepared_hash = digest(prepared)
            completed = {"status": "complete", "checks_pass": True, "prepared_hash": prepared_hash}
            write_json(root / "pilot-result.json", {**completed, "mode": "pilot"})
            selected = {"epoch": 1, "metrics": {"top3": .6}}
            write_json(root / "screen-result.json", {**completed, "mode": "screen", "selected_learning_rate": 1e-5, "selected": selected})
            rates = prepared["config"]["screen"]["learning_rates"]
            for rate in rates[:-1]:
                write_json(root / f"trials/screen/lr-{rate:g}/result.json", {**completed, "mode": "screen", "learning_rate": rate, "selected": selected})
            with self.assertRaises(FileNotFoundError):
                check_trial_policy(root, "full", 1e-5, prepared)
            write_json(root / f"trials/screen/lr-{rates[-1]:g}/result.json", {**completed, "mode": "screen", "learning_rate": rates[-1], "selected": selected})
            check_trial_policy(root, "full", 1e-5, prepared)
            with self.assertRaises(ValueError):
                check_trial_policy(root, "full", 4e-5, prepared)


if __name__ == "__main__":
    unittest.main()

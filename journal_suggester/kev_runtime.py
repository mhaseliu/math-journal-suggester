"""Construct native Kev training commands with pinned initialization."""

import sys
from pathlib import Path


def released(config):
    return config["kev_model"] + "@" + config["kev_revision"]


def training_command(
    directory,
    output,
    ck,
    config,
    max_steps=3,
    learning_rate=2e-5,
    batch=1,
    accum=8,
    checkpointing=True,
    preserve_candidate_order=False,
):
    entry = (
        "journal_suggester.ordered_train" if preserve_candidate_order else "kev.train"
    )
    return [
        sys.executable,
        "-m",
        "journal_suggester.gb10_runtime",
        entry,
        "--data",
        str(Path(directory) / "train.jsonl"),
        "--base",
        ck.meta.base,
        "--base_revision",
        ck.meta.base_revision,
        "--init_from",
        released(config),
        "--lora",
        str(ck.meta.lora),
        "--head_dim",
        str(ck.meta.head_dim),
        "--lora_targets",
        "all",
        "--option_isolation",
        str(int(ck.meta.option_isolation)),
        "--special_embeddings",
        str(int(ck.meta.special_embeddings)),
        "--epochs",
        "1",
        "--lr",
        str(learning_rate),
        "--batch",
        str(batch),
        "--accum",
        str(accum),
        "--dtype",
        "bf16",
        "--weights_dtype",
        "fp32",
        "--checkpointing",
        str(int(checkpointing)),
        "--seed",
        str(config["seed"]),
        "--device",
        "cuda",
        "--max_state",
        str(config["max_state"]),
        "--p_none",
        "0",
        "--p_none_distract",
        "0",
        "--p_distract",
        "0",
        "--p_none_pair",
        "0",
        "--perm_kl",
        "0",
        "--full_ft",
        "0",
        "--max_steps",
        str(max_steps),
        "--out",
        str(output),
    ]

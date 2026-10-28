#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["pandas>=2.2,<4", "bert-score==0.3.13", "transformers==4.44.2", "torch==2.4.1", "numpy<2"]
# ///
"""Add t0_bert through t3_bert to the merged CSV. Scores are F1 against T0.

python run_rq1_bertscore.py --device cuda
Repeat the command to resume from checkpoint.pkl. Increase --batch-size if GPU
memory allows. For parallel Slurm jobs, give each job a different --shard-index
and the same --num-shards. Each shard writes to its own output subdirectory.
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import signal
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TEXT_COLUMNS = ["t0", "t1", "t2", "t3"]
SCORE_COLUMNS = ["t0_bert", "t1_bert", "t2_bert", "t3_bert"]


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/organized_paraphrased_datasets/merged_paraphrased_wide.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/rq1_bertscore")
    parser.add_argument("--bert-model", default="roberta-large")
    parser.add_argument("--num-layers", type=positive_int, help="Override BERTScore's recommended layer count")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=positive_int, default=16, help="Pairs processed together on the GPU")
    parser.add_argument("--chunk-size", type=positive_int, default=128, help="Input rows processed before saving progress")
    parser.add_argument("--num-threads", type=positive_int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    parser.add_argument("--num-shards", type=positive_int, default=1, help="Split input into this many independent jobs")
    parser.add_argument("--shard-index", type=int, default=0, help="This job's shard, starting at 0")
    parser.add_argument("--max-rows", type=positive_int, help="Limit newly scored rows for a trial run")
    parser.add_argument("--max-minutes", type=positive_int, help="Stop between chunks after this many minutes")
    parser.add_argument("--dry-run", action="store_true", help="Count rows and valid pairs without loading a model")
    parser.add_argument("--export-only", action="store_true", help="Export checkpoint.pkl to scores.csv without loading a model")
    args = parser.parse_args(argv)
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be between 0 and --num-shards minus 1")
    return args


def read_data(args):
    """Read the original columns and add four numeric score columns."""
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    missing = set(TEXT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    if args.num_shards > 1:
        start = len(df) * args.shard_index // args.num_shards
        end = len(df) * (args.shard_index + 1) // args.num_shards
        df = df.iloc[start:end].copy()
    df[SCORE_COLUMNS] = np.nan
    # T0 compared with itself is the baseline. Missing references stay NaN.
    df.loc[df["t0"].str.strip().ne(""), "t0_bert"] = 1.0
    return df


def pending_pairs(df):
    """Make a temporary table of valid, unfinished candidate/reference pairs."""
    parts = []
    has_reference = df["t0"].str.strip().ne("")
    for step in ["t1", "t2", "t3"]:
        column = f"{step}_bert"
        unfinished = has_reference & df[step].str.strip().ne("") & df[column].isna()
        pairs = df.loc[unfinished, ["t0", step]].rename(columns={"t0": "reference", step: "candidate"})
        pairs["score_column"] = column
        parts.append(pairs)
    # Keep each row's three rounds together to reuse reference embeddings in a batch.
    return pd.concat(parts).sort_index(kind="stable")


def run_config(args):
    with args.input.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "input_sha256": digest, "model": args.bert_model, "num_layers": args.num_layers,
        "lang": "en", "idf": False, "rescale_with_baseline": False,
        "use_fast_tokenizer": False, "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }


@contextmanager
def output_lock(output_dir):
    """Allow one writer per output directory; different shards have separate locks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "checkpoint.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another job is writing to this output directory") from error
        yield


def atomic_save(df, path):
    """Keep the previous file intact if saving is interrupted."""
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        if path.suffix == ".pkl":
            df.to_pickle(temporary)
        else:
            df.to_csv(temporary, index=False)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def restore_progress(df, path, config):
    if not path.exists():
        return None
    saved = pd.read_pickle(path)
    if saved.attrs.get("config") != config or not saved.index.equals(df.index):
        raise ValueError("Checkpoint input or settings changed. Use the original settings or a new --output-dir.")
    df[SCORE_COLUMNS] = saved[SCORE_COLUMNS]
    print(f"Restored scores from {path}", flush=True)
    return saved.attrs.get("runtime")


def save_progress(df, path, config, runtime):
    # Save only the four score columns between chunks, rather than copying all text.
    saved = df[SCORE_COLUMNS].copy()
    saved.attrs = {"config": config, "runtime": runtime}
    atomic_save(saved, path)


def load_scorer(args):
    import torch
    from bert_score import BERTScorer

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Request a GPU before running with --device cuda.")
    torch.set_num_threads(args.num_threads)
    print(f"Loading {args.bert_model} on {device}; CPU threads: {args.num_threads}", flush=True)
    scorer = BERTScorer(
        model_type=args.bert_model, num_layers=args.num_layers, lang="en",
        device=device, batch_size=args.batch_size, nthreads=args.num_threads,
        idf=False, rescale_with_baseline=False, use_fast_tokenizer=False,
    )
    runtime = {name: version(name) for name in ["bert-score", "transformers", "torch", "tokenizers", "numpy", "pandas"]}
    runtime["bertscore_hash"] = scorer.hash
    return scorer, runtime


def score_chunk(df, pairs, scorer, batch_size):
    """Score several rows' rounds together, then write F1 into the wide table."""
    _, _, f1 = scorer.score(
        cands=pairs["candidate"].tolist(), refs=pairs["reference"].tolist(),
        batch_size=batch_size,
    )
    scores = f1.tolist()
    if len(scores) != len(pairs) or not np.isfinite(scores).all():
        raise ValueError("BERTScore returned missing or non-finite scores; this chunk was not saved.")
    pairs = pairs.assign(score=scores)
    for column, group in pairs.groupby("score_column", sort=False):
        df.loc[group.index, column] = group["score"]


def run(args, stop_requested=lambda: False):
    started = time.monotonic()

    # 1. Read the wide CSV with pandas and add t0_bert ... t3_bert.
    df = read_data(args)
    output_dir = args.output_dir
    if args.num_shards > 1:
        output_dir = output_dir / f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}"
    if args.dry_run:
        print(f"Rows: {len(df):,}; valid T1-T3 pairs: {len(pending_pairs(df)):,}")
        return
    checkpoint = output_dir / "checkpoint.pkl"
    if args.export_only and not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint found: {checkpoint}")
    config = run_config(args)
    with output_lock(output_dir):
        runtime = restore_progress(df, checkpoint, config)
        if args.export_only:
            atomic_save(df, output_dir / "scores.csv")
            return

        # 2. Select unfinished rows. Missing text stays NaN and is never scored.
        pairs = pending_pairs(df)
        row_indices = pairs.index.unique()
        if args.max_rows:
            row_indices = row_indices[:args.max_rows]
        print(f"Unfinished rows this run: {len(row_indices):,}", flush=True)
        scorer = None

        # 3. Batch pairs on the GPU and save a pandas checkpoint after each chunk.
        try:
            for start in range(0, len(row_indices), args.chunk_size):
                if stop_requested() or (args.max_minutes and time.monotonic() - started >= args.max_minutes * 60):
                    break
                if scorer is None:
                    scorer, current_runtime = load_scorer(args)
                    if runtime is not None and runtime != current_runtime:
                        raise ValueError("Scoring package versions changed. Restore the environment or use a new --output-dir.")
                    runtime = current_runtime
                    if stop_requested() or (args.max_minutes and time.monotonic() - started >= args.max_minutes * 60):
                        break
                rows = row_indices[start:start + args.chunk_size]
                score_chunk(df, pairs.loc[rows], scorer, args.batch_size)
                save_progress(df, checkpoint, config, runtime)
                print(f"Saved {min(start + args.chunk_size, len(row_indices)):,}/{len(row_indices):,} rows", flush=True)
        finally:
            # Full text is written once on exit. SIGKILL still leaves checkpoint.pkl intact.
            atomic_save(df, output_dir / "scores.csv")
            print(f"Wide CSV: {output_dir / 'scores.csv'}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    stopped = False

    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True

    for signum in [signal.SIGINT, signal.SIGTERM, signal.SIGUSR1]:
        signal.signal(signum, request_stop)
    run(args, lambda: stopped)


if __name__ == "__main__":
    main()

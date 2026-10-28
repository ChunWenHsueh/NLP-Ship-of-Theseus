"""Compute T0-vs-T1/T2/T3 BERTScore F1, saving progress every 1,000 pairs."""

import argparse
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"  # BERTScore imports Matplotlib, including in Colab.
import numpy as np
import pandas as pd
import torch
from bert_score import BERTScorer

ROOT = Path(__file__).resolve().parent
TEXT_COLUMNS = ["t0", "t1", "t2", "t3"]
SCORE_COLUMNS = [f"{step}_bert" for step in TEXT_COLUMNS]


def save_checkpoint(df, path):
    temporary = path.with_suffix(".tmp")
    df.to_pickle(temporary)
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT
        / "data/organized_paraphrased_datasets/merged_paraphrased_wide.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/rq1_bertscore"
    )
    parser.add_argument("--bert-model", default="roberta-large")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=256)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.save_every < 1:
        parser.error("--batch-size and --save-every must be positive")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "checkpoint.pkl"
    settings = {"model": args.bert_model, "num_layers": None}

    # Prefer the complete checkpoint; only a fresh run needs the input CSV.
    if checkpoint.exists():
        df = pd.read_pickle(checkpoint)
        old_settings = df.attrs.get("config", {})
        if any(old_settings.get(k) != v for k, v in settings.items()):
            raise ValueError(
                "Checkpoint model settings differ. Use a new --output-dir."
            )
        # Older checkpoints contain only scores; add the text once to migrate them.
        if not set(TEXT_COLUMNS).issubset(df.columns):
            saved = df
            df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
            if not saved.index.equals(df.index):
                raise ValueError("Old checkpoint rows differ from the input CSV")
            df[SCORE_COLUMNS] = saved[SCORE_COLUMNS]
            df.attrs = saved.attrs
            save_checkpoint(df, checkpoint)
        print(f"Resumed from {checkpoint}", flush=True)
    else:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        df[SCORE_COLUMNS] = np.nan
    has_reference = df["t0"].str.strip().ne("")
    df.loc[has_reference & df["t0_bert"].isna(), "t0_bert"] = 1.0
    df.attrs["config"] = settings

    # Load the model once, then compare each step directly with T0.
    scorer = None
    for step in ["t1", "t2", "t3"]:
        column = f"{step}_bert"
        pending = has_reference & df[step].str.strip().ne("") & df[column].isna()
        indices = df.index[pending]
        print(f"{step}: {len(indices):,} pairs remaining", flush=True)
        for start in range(0, len(indices), args.save_every):
            if scorer is None:
                print(f"Loading {args.bert_model} on {device}...", flush=True)
                scorer = BERTScorer(
                    model_type=args.bert_model,
                    lang="en",
                    device=device,
                    idf=False,
                    rescale_with_baseline=False,
                    use_fast_tokenizer=False,
                )
                print(f"Model ready, layer {scorer.num_layers}", flush=True)
            rows = indices[start : start + args.save_every]
            _, _, f1 = scorer.score(
                cands=df.loc[rows, step].tolist(),
                refs=df.loc[rows, "t0"].tolist(),
                batch_size=args.batch_size,
                verbose=True,
            )
            scores = f1.tolist()
            if len(scores) != len(rows) or not np.isfinite(scores).all():
                raise ValueError("BERTScore returned missing or non-finite scores")
            df.loc[rows, column] = scores

            # Save the entire wide table so the next run needs only this file.
            save_checkpoint(df, checkpoint)
            print(
                f"{step}: saved {min(start + args.save_every, len(indices)):,}/{len(indices):,} pairs",
                flush=True,
            )

    output = args.output_dir / "scores.csv"
    df.to_csv(output, index=False)
    print(f"Completed. Wide CSV: {output}", flush=True)


if __name__ == "__main__":
    main()

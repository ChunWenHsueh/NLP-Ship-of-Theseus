import argparse
from pathlib import Path
import pandas as pd

METRICS = ["bleu", "rouge_l", "bertscore_f1"]

def parse_args():
    parser = argparse.ArgumentParser(description="Create T0-T3 similarity comparison tables")

    script_dir = Path(__file__).resolve().parent

    parser.add_argument("--input", type=Path, default=script_dir / "similarity_scores_full.csv")
    parser.add_argument("--out", type=Path, default=script_dir / "metrics_table.csv")

    return parser.parse_args()

def main():
    args = parse_args()

    # load similarity scores.
    df = pd.read_csv(args.input)

    # check required columns
    required = {"dataset", "paraphraser", "round", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # compute mean scores for each dataset, paraphraser, and round
    summary = (
        df.groupby(["dataset", "paraphraser", "round"])[METRICS]
        .mean()
        .round(3)
        .reset_index()
    )

    # label paraphrasing rounds as T1, T2, and T3
    summary["round"] = summary["round"].apply(
        lambda r: f"T{int(r)}"
    )

    tables = []

    # display a comparison table for each dataset
    for dataset in summary["dataset"].unique():
        sub = summary[summary["dataset"] == dataset]

        # add T0 as the original reference baseline
        baseline = pd.DataFrame([{
            "dataset": dataset,
            "paraphraser": "Original",
            "round": "T0",
            "bleu": 1.0,
            "rouge_l": 1.0,
            "bertscore_f1": float("nan")
        }])

        table = pd.concat([baseline, sub], ignore_index=True)
        tables.append(table)

        print()
        print(f"Dataset: {dataset}")
        print(table.to_string(index=False))

    # save comparison tables to CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(args.out, index=False)
    print()
    print(f"Metrics table saved")


if __name__ == "__main__":
    main()

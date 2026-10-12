import argparse
from pathlib import Path
import pandas as pd

DATASETS = ["xsum", "yelp"]
PARAPHRASERS = ["chatgpt", "palm", "dipper", "pegasus(full)"]

# load paraphrased datasets
def load_corpus(data_dir):
    dfs = {}
    for ds in DATASETS:
        path = data_dir / "paraphrased_datasets" / f"{ds}_paraphrased.csv"
        if not path.exists():
            raise FileNotFoundError(f"Expected file not found: {path}")
        dfs[ds] = pd.read_csv(path)
    return dfs

# mark the version of each paraphraser
def parse_version(version_name):
    if version_name == "original":
        return "none", 0
    parts = version_name.split("_")
    paraphraser = parts[0] # first segment is the paraphraser
    round_n = len(parts)   # number of segments is the round
    return paraphraser, round_n

# match each paraphrased text (T1-T3) with its original text (T0)
def build_pairs(df, dataset_name):
    df = df.copy()
    parsed = df["version_name"].apply(parse_version)
    df["paraphraser"] = parsed.apply(lambda t: t[0])
    df["round"] = parsed.apply(lambda t: t[1])

    # select original texts (T0)
    originals = (
        df[df["round"] == 0][["key", "source", "text"]]
        .rename(columns={"text": "original_text"})
    )

    # select T1-T3 from the four target paraphrasers
    paraphrased = df[
        df["paraphraser"].isin(PARAPHRASERS) & df["round"].isin([1, 2, 3])
    ][["key", "source", "paraphraser", "round", "text"]].rename(
        columns={"text": "paraphrased_text"}
    )

     # match by both key and source
    pairs = paraphrased.merge(originals, on=["key", "source"], how="inner")
    pairs.insert(0, "dataset", dataset_name)
    return pairs

# parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Build T0 vs T1-T3 pairs for XSum and Yelp")
    project_root = Path(__file__).resolve().parents[2]

    parser.add_argument("--data-dir", type=Path, default=project_root / "data")
    parser.add_argument("--out", type=Path, default=project_root / "results" / "paired_table.csv")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    dfs = load_corpus(args.data_dir)

    # build T0 vs T1-T3 pairs for each dataset
    all_pairs = pd.concat([build_pairs(dfs[ds], ds) for ds in DATASETS], ignore_index=True)

    print("Total pairs:", len(all_pairs))
    print()
    print("Pairs per (dataset, paraphraser, round):")
    print(all_pairs.groupby(["dataset", "paraphraser", "round"]).size().unstack(fill_value=0))
    print()
    preview = all_pairs.head(3).copy()
    preview["original_text"] = preview["original_text"].str.slice(0, 60)
    preview["paraphrased_text"] = preview["paraphrased_text"].str.slice(0, 60)
    print(preview[["dataset", "key", "source", "paraphraser", "round", "original_text", "paraphrased_text"]].to_string(index=False))

    all_pairs.to_csv("paired_table.csv", index=False)
    print()
    print("paired_table.csv saved")
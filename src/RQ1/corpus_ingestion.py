import argparse
import re
from pathlib import Path
import pandas as pd

DATASETS = ["xsum", "yelp"]

# load csv file for each dataset
def load_corpus(data_dir):
    dfs = {}
    for ds in DATASETS:
        train_path = data_dir / "train_datasets" / f"{ds}_train.csv"
        test_path = data_dir / "paraphrased_datasets" / f"{ds}_paraphrased.csv"
        for key, path in [(f"{ds}_train", train_path), (f"{ds}_test", test_path)]:
            if not path.exists():
                raise FileNotFoundError(f"Expected file not found: {path}")
            dfs[key] = pd.read_csv(path)
    return dfs

# mark the version of each paraphraser
def parse_version(version_name):
    if version_name == "original":
        return "none", 0
    parts = version_name.split("_")
    paraphraser = parts[0] # first segment is the paraphraser
    round_n = len(parts)   # number of segments is the round
    return paraphraser, round_n

# derive paraphraser and round columns from version_name
def add_version_columns(df):
    df = df.copy()
    parsed = df["version_name"].apply(parse_version)
    df["paraphraser"] = parsed.apply(lambda t: t[0])
    df["round"] = parsed.apply(lambda t: t[1])
    return df

def word_count(text):
    return len(str(text).split())

# Count total words and unique words in a fixed-size sample, 
# and then equaly compare xsum and yelp
def vocab_size(texts, sample_n = 200, seed = 42):
    texts = texts.dropna()
    sample = texts.sample(min(sample_n, len(texts)), random_state=seed)
    tokens = []
    for t in sample:
        tokens.extend(re.findall(r"\w+", str(t).lower()))
    return {
        "n_docs_sampled": len(sample),
        "total_words": len(tokens),
        "unique_words": len(set(tokens)),
    }

# Summary statistics for the train subset.
def summarize_train(name, df):
    lengths = df["text"].apply(word_count)
    human_texts = df.loc[df["source"] == "Human", "text"]
    vocab = vocab_size(human_texts)
    return {
        "dataset": name,
        "n_rows": len(df),
        "n_unique_articles": df["key"].nunique(),
        "n_sources": df["source"].nunique(),
        "mean_words": round(lengths.mean(), 1),
        "median_words": int(lengths.median()),
        "max_words": int(lengths.max()),
        "human_total_words": vocab["total_words"],
        "human_unique_words": vocab["unique_words"],
    }

# Summary statistics for paraphrased subset.
def summarize_test(name, df):
    return {
        "dataset": name,
        "n_rows": len(df),
        "n_unique_articles": df["key"].nunique(),
        "n_sources": df["source"].nunique(),
        "n_versions": df["version_name"].nunique(),
        "max_round": int(df["round"].max()),
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Ship of Theseus corpus ingestion (xsum/yelp)")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
 
    dfs = load_corpus(args.data_dir)
 
    for ds in DATASETS:
        dfs[f"{ds}_test"] = add_version_columns(dfs[f"{ds}_test"])
 
    print()
    print("TRAIN SET SUMMARY")
    train_rows = [summarize_train(f"{ds}_train", dfs[f"{ds}_train"]) for ds in DATASETS]
    train_summary = pd.DataFrame(train_rows)
    print(train_summary.to_string(index=False))
 
    print()
    print("TEST SET SUMMARY")
    test_rows = [summarize_test(f"{ds}_test", dfs[f"{ds}_test"]) for ds in DATASETS]
    test_summary = pd.DataFrame(test_rows)
    print(test_summary.to_string(index=False))
    
    print()
    print("SOURCE DISTRIBUTION")
    for ds in DATASETS:
        name = f"{ds}_train"
        print(f"--- {name} ---")
        print(dfs[name]["source"].value_counts())
        print()
 
    print("VERSION / ROUND DISTRIBUTION ")
    for ds in DATASETS:
        name = f"{ds}_test"
        print(f"--- {name}: rows per (paraphraser, round) ---")
        print(dfs[name].groupby(["paraphraser", "round"]).size().unstack(fill_value=0))
        print()
 
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        train_summary.to_csv(args.out, index=False)
        print(f"Train summary written to: {args.out}")
 
 
if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import pandas as pd

# read similarity scores
df = pd.read_csv("similarity_scores_full.csv")

PARAPHRASERS = ["chatgpt", "palm", "dipper", "pegasus(full)"]
METRICS = ["bleu", "rouge_l", "bertscore_f1"]
DATASETS = ["xsum", "yelp"]

# select the four paraphrasers and T1-T3.
df = df[
    df["paraphraser"].isin(PARAPHRASERS)
    & df["round"].isin([1, 2, 3])
]

# calculate mean similarity for each dataset, paraphraser, and round
agg = (
    df.groupby(["dataset", "paraphraser", "round"])[METRICS]
    .mean()
    .reset_index()
)

# create a figure: datasets as rows and metrics as columns
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

METRIC_LABELS = {
    "bleu": "BLEU (Lexical)",
    "rouge_l": "ROUGE-L (Lexical)",
    "bertscore_f1": "BERTScore F1 (Semantic)",
}

# plot similarity decay for each dataset and metric
for i, dataset in enumerate(DATASETS):
    for j, metric in enumerate(METRICS):
        ax = axes[i, j]

        # plot one line for each paraphraser
        for paraphraser in PARAPHRASERS:
            sub = agg[
                (agg["dataset"] == dataset)
                & (agg["paraphraser"] == paraphraser)
            ].sort_values("round")

            ax.plot(
                sub["round"],
                sub[metric],
                marker="o",
                label=paraphraser,
            )

        # configure subplot titles, axes, and grid
        ax.set_title(f"{dataset.upper()} - {METRIC_LABELS[metric]}")
        ax.set_xlabel("Paraphrase round")
        ax.set_ylabel("Similarity to T0")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["T1", "T2", "T3"])
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

# add a shared legend and figure title
handles, labels = axes[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, 0.02),
)

fig.suptitle(
    "Lexical vs. Semantic Similarity Across Paraphrasing Rounds",
    fontsize=14,
)

fig.tight_layout(rect=[0, 0.07, 1, 0.95])
fig.savefig("style_vs_content_decay.png", dpi=150)

print("Picture saved")

import matplotlib.pyplot as plt
import pandas as pd

# read file
df = pd.read_csv("similarity_scores.csv")

# calculate mean across all four paraphrasers
agg = df.groupby(["dataset", "round"])[["bleu", "rouge_l", "bertscore_f1"]].mean()

# create a subplot for dataset
datasets = df["dataset"].unique()
fig, axes = plt.subplots(1, len(datasets), figsize=(12, 5), sharey=True)

for ax, dataset in zip(axes, datasets):
    sub = agg.loc[dataset]

    # add T0 baseline
    rounds = [0] + sub.index.tolist()
    bleu_vals = [1.0] + sub["bleu"].tolist()
    rouge_vals = [1.0] + sub["rouge_l"].tolist()
    bert_vals = [1.0] + sub["bertscore_f1"].tolist()

    ax.plot(rounds, bleu_vals, marker="o", label="BLEU (lexical / plank)", color="#d62728")
    ax.plot(rounds, rouge_vals, marker="s", label="ROUGE-L (lexical / plank)", color="#ff7f0e")
    ax.plot(rounds, bert_vals, marker="^", label="BERTScore (semantic / hull)", color="#1f77b4", linewidth=2.5)

    # subplot titles, axes, and grid.
    ax.set_title(f"{dataset.upper()}")
    ax.set_xlabel("Paraphrase round")
    ax.set_xticks(rounds)
    ax.set_xticklabels([f"T{r}" for r in rounds])
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

# add the shared y-axis label
axes[0].set_ylabel("Similarity to T0")
axes[0].legend(loc="lower left", fontsize=9)

fig.suptitle("style vs content_decay", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("style_vs_content_decay.png", dpi=150)
print("Picture saved")
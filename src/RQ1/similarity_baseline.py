import argparse

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bertscore

CHENCHERRY = SmoothingFunction().method1

# compute BLEU score between the original and paraphrased text
def compute_bleu(reference, candidate):
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=CHENCHERRY)

# compute ROUGE score based on longest common subsequence
def compute_rouge_l(scorer, reference, candidate):
    return scorer.score(reference, candidate)["rougeL"].fmeasure

# randomly sample up to n pairs for each dataset, paraphraser, and round
def sample_per_cell(df, n: int, seed: int = 42):
    return (
        df.groupby(["dataset", "paraphraser", "round"], group_keys=False)
        .apply(lambda g: g.sample(min(n, len(g)), random_state=seed))
        .reset_index(drop=True)
    )

def main():
    parser = argparse.ArgumentParser(description="Compute similarity metrics for T0 vs T1/T2/T3 pairs")
    parser.add_argument("--pairs-file", default="paired_table.csv")
    parser.add_argument("--sample-per-cell", type=int, default=200)
    parser.add_argument("--bert-model", default="roberta-large")
    parser.add_argument("--out", default="similarity_scores.csv")
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs_file)
    pairs = sample_per_cell(pairs, args.sample_per_cell)
    print(f"Scoring {len(pairs)} sampled pairs...")

    # ompute lexical similarity using BLEU and ROUGE
    rouge_scorer_obj = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    pairs["bleu"] = pairs.apply(
        lambda r: compute_bleu(r["original_text"], r["paraphrased_text"]), axis=1
    )
    pairs["rouge_l"] = pairs.apply(
        lambda r: compute_rouge_l(rouge_scorer_obj, r["original_text"], r["paraphrased_text"]), axis=1
    )

    # compute semantic similarity using BERTScore
    P, R, F1 = bertscore(
        cands=pairs["paraphrased_text"].tolist(),
        refs=pairs["original_text"].tolist(),
        lang="en",
        model_type=args.bert_model,
        verbose=True,
    )
    pairs["bertscore_f1"] = F1.tolist()
    pairs.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
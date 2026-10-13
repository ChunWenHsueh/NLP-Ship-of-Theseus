import argparse
from pathlib import Path
import pandas as pd
import torch
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

def main():
    parser = argparse.ArgumentParser(description="Compute similarity metrics for T0 vs T1/T2/T3 pairs")
    parser.add_argument("--pairs-file", default="paired_table.csv")
    parser.add_argument("--bert-model", default="roberta-large")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out",type=Path, default=Path("results/similarity_scores_full.csv"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pairs = pd.read_csv(args.pairs_file)
    print(f"Scoring {len(pairs)} pairs")

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
        device=device,
        batch_size=args.batch_size,
        verbose=True,
    )
    pairs["bertscore_f1"] = F1.tolist()
    print("BERTScore completed")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.out, index=False)
    print(f"Similarity scores saved")

if __name__ == "__main__":
    main()
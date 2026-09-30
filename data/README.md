# Ship of Theseus Dataset Guide

The data in this directory comes from **[A Ship of Theseus: Curious Cases of Paraphrasing in LLM-Generated Texts](https://aclanthology.org/2024.acl-long.357/)** by Tripto et al., published at **ACL 2024**. The authors released the corpus in their [official dataset repository](https://github.com/tripto03/Ship_of_theseus_paraphrased_copus).

## 1. Directory structure

```text
data/
├── README.md
├── organize_paraphrased.py
├── train_datasets/                    # downloaded; ignored by Git
│   ├── cmv_train.csv
│   ├── eli5_train.csv
│   ├── sci_gen_train.csv
│   ├── tldr_train.csv
│   ├── wp_train.csv
│   ├── xsum_train.csv
│   └── yelp_train.csv
└── paraphrased_datasets/             # downloaded; ignored by Git
    ├── cmv_paraphrased.csv
    ├── eli5_paraphrased.csv
    ├── sci_gen_paraphrased.csv
    ├── tldr_paraphrased.csv
    ├── wp_paraphrased.csv
    ├── xsum_paraphrased.csv
    └── yelp_paraphrased.csv
```

On a fresh clone, download the corpus from the project root:

```bash
corpus_tmp=$(mktemp -d)
git clone --depth 1 https://github.com/tripto03/Ship_of_theseus_paraphrased_copus.git "$corpus_tmp/corpus"
mv "$corpus_tmp/corpus/paraphrased_datasets" "$corpus_tmp/corpus/train_datasets" data/
rm -rf "$corpus_tmp"
```

The downloaded dataset directories and generated files are ignored by this
project's Git repository.

## 2. Organize paraphrasing chains

From the repository root, run:

```bash
uv run python data/organize_paraphrased.py
```

This creates one CSV per dataset in `data/organized_paraphrased_datasets/`, with columns
`source,key,paraphraser,t0,t1,t2,t3`. Each `(source, key)` has one row for each of the
seven paraphraser configurations. `t0` is the `original` text, and `t1`–`t3` are
successive rewrites by the named paraphraser. Missing versions are empty cells,
including a missing `original`.

## 3. CSV columns

Each of the seven datasets has a training file and an evaluation file:

| Directory               | Contents                                                            | Purpose                                  |
| ----------------------- | ------------------------------------------------------------------- | ---------------------------------------- |
| `train_datasets/`       | Unmodified texts from human and LLM sources                         | Train original-source classifiers        |
| `paraphrased_datasets/` | A separate set of original texts (T0) and their paraphrases (T1–T3) | Analyze changes and evaluate classifiers |

## 4. CSV columns

Column names are lowercase. Training files contain `source`, `key`, and `text`; evaluation files also contain `version_name`.

| Column         | Meaning                                                                                     | Example                           |
| -------------- | ------------------------------------------------------------------------------------------- | --------------------------------- |
| `source`       | The source that produced T0; this label remains unchanged after paraphrasing                | `Human`, `BigScience`, `PaLM`     |
| `key`          | The originating article ID, shared by human and model-generated texts based on that article | `yelp-494`                        |
| `text`         | The complete text for this particular record and version                                    | An original or a paraphrased text |
| `version_name` | The paraphraser/configuration and iteration; `original` denotes T0                          | `original`, `chatgpt_chatgpt`     |

## 5. Where does T0 come from?

T0 is a text before any paraphrasing. The researchers kept human-written articles and prompted six LLMs to continue the opening of each article (approximately 30 tokens). Each human article and each model-generated continuation is a separate T0, sharing the same `key` but having a different `source`.

## 6. Original authors / sources

The corpus has seven source categories.

| `source`      | Original source/model   |
| ------------- | ----------------------- |
| `Human`       | Human-written article   |
| `OpenAI`      | ChatGPT / GPT-3.5-turbo |
| `PaLM`        | Google PaLM 2           |
| `LLAMA`       | LLaMA-65B               |
| `BigScience`  | BLOOM-7B1               |
| `Eleuther-AI` | GPT-NeoX-20B            |
| `Tsinghua`    | GLM-130B                |

## 7. Paraphrasers

Four model families provide seven paraphrasing configurations:

| Model   | Name in `version_name`                  | Configuration                     |
| ------- | --------------------------------------- | --------------------------------- |
| ChatGPT | `chatgpt`                               | Full-text paraphrasing            |
| PaLM 2  | `palm`                                  | Full-text paraphrasing            |
| Dipper  | `dipper`, `dipper(low)`, `dipper(high)` | Default, low, or high diversity   |
| Pegasus | `pegasus(full)`, `pegasus(slight)`      | All sentences or 25% of sentences |

Each configuration starts from T0 and rewrites its own previous output: **T0 → T1 → T2 → T3**. For example, `chatgpt_chatgpt` means two successive ChatGPT rewrites.

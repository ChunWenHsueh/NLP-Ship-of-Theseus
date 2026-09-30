"""Turn each paraphrasing chain into one CSV row with T0 through T3."""

import argparse
import csv
import os
import tempfile
from pathlib import Path


PARAPHRASERS = (
    "chatgpt",
    "palm",
    "dipper",
    "dipper(low)",
    "dipper(high)",
    "pegasus(full)",
    "pegasus(slight)",
)
VERSION_TO_CHAIN = {
    "_".join([paraphraser] * iteration): (paraphraser, f"t{iteration}")
    for paraphraser in PARAPHRASERS
    for iteration in (1, 2, 3)
}
OUTPUT_COLUMNS = ("source", "key", "paraphraser", "t0", "t1", "t2", "t3")
INPUT_COLUMNS = {"source", "key", "text", "version_name"}


def organize_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    # Dicts retain input order, so output order is stable across runs.
    records: dict[tuple[str, str], dict[str, object]] = {}

    with input_path.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        if reader.fieldnames is None or not INPUT_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(f"{input_path}: expected columns {sorted(INPUT_COLUMNS)}")

        for row in reader:
            source, key = row["source"], row["key"]
            if not source or not key:
                raise ValueError(
                    f"{input_path}, line {reader.line_num}: empty source or key"
                )

            identity = (source, key)
            record = records.setdefault(identity, {"original": None, "chains": {}})
            version = row["version_name"]
            text = row["text"]

            if version == "original":
                if record["original"] is not None:
                    raise ValueError(
                        f"{input_path}, line {reader.line_num}: duplicate original for {identity}"
                    )
                record["original"] = text
                continue

            if version not in VERSION_TO_CHAIN:
                raise ValueError(
                    f"{input_path}, line {reader.line_num}: unknown version {version!r}"
                )
            paraphraser, column = VERSION_TO_CHAIN[version]
            chains = record["chains"]
            chain = chains.setdefault(paraphraser, {})
            if column in chain:
                raise ValueError(
                    f"{input_path}, line {reader.line_num}: duplicate {version} for {identity}"
                )
            chain[column] = text

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for (source, key), record in records.items():
                chains = record["chains"]
                for paraphraser in PARAPHRASERS:
                    chain = chains.get(paraphraser, {})
                    writer.writerow(
                        {
                            "source": source,
                            "key": key,
                            "paraphraser": paraphraser,
                            "t0": record["original"]
                            if record["original"] is not None
                            else "",
                            "t1": chain.get("t1", ""),
                            "t2": chain.get("t2", ""),
                            "t3": chain.get("t3", ""),
                        }
                    )
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    missing_originals = sum(record["original"] is None for record in records.values())
    return len(records) * len(PARAPHRASERS), missing_originals


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=base / "paraphrased_datasets",
        help="Directory containing *_paraphrased.csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "organized_paraphrased_datasets",
        help="Directory for the seven organized CSV files",
    )
    args = parser.parse_args()

    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("input and output directories must differ")
    input_paths = sorted(args.input_dir.glob("*_paraphrased.csv"))
    if not input_paths:
        parser.error(
            f"no *_paraphrased.csv files found in {args.input_dir}; "
            "see data/README.md for download instructions"
        )

    for input_path in input_paths:
        output_path = args.output_dir / f"{input_path.stem}_wide.csv"
        row_count, missing_originals = organize_file(input_path, output_path)
        print(
            f"{input_path.name} -> {output_path} ({row_count} rows; "
            f"{missing_originals} source/key groups missing T0)"
        )


if __name__ == "__main__":
    main()

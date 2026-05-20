import argparse
import logging
import numbers
from typing import Iterable, List

import numpy as np
import pandas as pd


def _normalize_token(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _coerce_binary_numeric(value):
    if isinstance(value, (numbers.Number, np.generic)):
        if value in {0, 1}:
            return int(value)
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"0", "1"}:
            return int(token)
        try:
            parsed = float(token)
        except ValueError:
            return None
        if parsed in {0.0, 1.0}:
            return int(parsed)
    return None


def _parse_list(values: str | None) -> List[str]:
    if not values:
        return []
    return [v.strip().lower() for v in values.split(",") if v.strip()]


def normalize_labels(
    df: pd.DataFrame,
    source_column: str,
    target_column: str,
    positive: Iterable[str],
    negative: Iterable[str],
    drop_unmapped: bool = True,
) -> pd.DataFrame:
    pos_set = {v.lower() for v in positive}
    neg_set = {v.lower() for v in negative}

    labels = []
    keep_mask = []
    for val in df[source_column].tolist():
        token = _normalize_token(val)
        mapped = _coerce_binary_numeric(token)
        if mapped is None:
            if token in pos_set:
                mapped = 1
            elif token in neg_set:
                mapped = 0

        if mapped is None:
            keep_mask.append(not drop_unmapped)
            labels.append(mapped)
        else:
            keep_mask.append(True)
            labels.append(mapped)

    out = df.copy()
    out[target_column] = labels
    if drop_unmapped:
        out = out[keep_mask]
    return out


def main(
    input_file: str,
    output_file: str,
    source_column: str,
    target_column: str,
    positive: str | None,
    negative: str | None,
    drop_unmapped: bool,
) -> None:
    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv(input_file)
    if source_column not in df.columns:
        raise ValueError(f"Missing source label column '{source_column}'.")

    pos_list = _parse_list(positive)
    neg_list = _parse_list(negative)
    if not pos_list or not neg_list:
        raise ValueError("Positive and negative label lists must be provided.")

    out = normalize_labels(
        df,
        source_column=source_column,
        target_column=target_column,
        positive=pos_list,
        negative=neg_list,
        drop_unmapped=drop_unmapped,
    )
    out.to_csv(output_file, index=False)
    logging.info("Wrote %s rows to %s", len(out), output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize string labels to 0/1.")
    parser.add_argument("input_file", type=str, help="Input CSV path.")
    parser.add_argument("output_file", type=str, help="Output CSV path.")
    parser.add_argument("--source-column", type=str, required=True, help="Source label column.")
    parser.add_argument("--target-column", type=str, required=True, help="Target label column.")
    parser.add_argument("--positive", type=str, required=True, help="Comma-separated positive labels.")
    parser.add_argument("--negative", type=str, required=True, help="Comma-separated negative labels.")
    parser.add_argument(
        "--drop-unmapped",
        action="store_true",
        help="Drop rows with labels not in positive/negative lists.",
    )
    args = parser.parse_args()

    main(
        args.input_file,
        args.output_file,
        source_column=args.source_column,
        target_column=args.target_column,
        positive=args.positive,
        negative=args.negative,
        drop_unmapped=args.drop_unmapped,
    )

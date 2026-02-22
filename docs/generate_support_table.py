#!/usr/bin/env python3
from __future__ import annotations

import html
import re
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent

PROFILE_HEADERS = [
    "Task Type",
    "Profile",
    "Typical Datasets",
    "get_data",
    "curate",
    "labels",
    "split",
    "featurize/use",
    "preprocess + select",
    "train",
    "train.tdc",
    "analyze",
    "explain",
]

PROFILE_ROWS = [
    {
        "Task Type": "Regression",
        "Profile": "`reg_local_csv`",
        "Typical Datasets": "flash, qm9, ysi, pah",
        "get_data": "`local_csv`",
        "curate": "`properties` (+ optional `smiles_column`, `dedupe_strategy`, `keep_all_columns`)",
        "labels": "",
        "split": "`holdout / cv / nested_holdout_cv`<br>`random / scaffold`",
        "featurize/use": "`use.curated_features`, `featurize.rdkit`, `featurize.morgan`",
        "preprocess + select": "Yes (feature input required; `select.features` requires `preprocess.features`)",
        "train": "`random_forest`, `svm`, `decision_tree`, `xgboost`, `ensemble`, `dl_*`",
        "train.tdc": "",
        "analyze": "",
        "explain": "Yes",
    },
    {
        "Task Type": "Regression",
        "Profile": "`reg_chembl_ic50`",
        "Typical Datasets": "chembl (urease regression)",
        "get_data": "`chembl`",
        "curate": "`properties: standard_value`, `keep_all_columns`",
        "labels": "`label.ic50`",
        "split": "`holdout / cv / nested_holdout_cv`<br>`random / scaffold`",
        "featurize/use": "`featurize.lipinski` + `featurize.rdkit` (rdkit input required)",
        "preprocess + select": "Yes (feature input required; `select.features` requires `preprocess.features`)",
        "train": "`random_forest`, `svm`, `decision_tree`, `xgboost`, `ensemble`, `dl_*`",
        "train.tdc": "",
        "analyze": "`stats`, `eda`",
        "explain": "Yes",
    },
    {
        "Task Type": "Classification",
        "Profile": "`clf_local_csv`",
        "Typical Datasets": "pgp, ara",
        "get_data": "`local_csv`",
        "curate": "`properties`, `smiles_column`, `dedupe_strategy`, `label_column`",
        "labels": "Optional `label.normalize` (required if target is not already binary)",
        "split": "`holdout / cv / nested_holdout_cv`<br>`random / scaffold`",
        "featurize/use": "`catboost_classifier`/`dl_*`: one of `use.curated_features`, `featurize.rdkit`, `featurize.morgan`<br>`chemprop`: no tabular feature node required",
        "preprocess + select": "Yes for non-chemprop (requires feature input). Not supported with `chemprop`.",
        "train": "`catboost_classifier`, `chemprop`, `dl_*`",
        "train.tdc": "",
        "analyze": "",
        "explain": "Yes* (chemprop explain is skipped)",
    },
    {
        "Task Type": "Classification",
        "Profile": "`clf_tdc_benchmark`",
        "Typical Datasets": "pgp_tdc_benchmark",
        "get_data": "",
        "curate": "",
        "labels": "",
        "split": "",
        "featurize/use": "",
        "preprocess + select": "",
        "train": "",
        "train.tdc": "`catboost_classifier`",
        "analyze": "",
        "explain": "",
    },
]

DATASET_HEADERS = [
    "Dataset (Profile)",
    "get_data",
    "curate",
    "label.normalize",
    "label.ic50",
    "split",
    "featurize / use",
    "preprocess.features",
    "select.features",
    "train",
    "train.tdc",
    "analyze.stats",
    "analyze.eda",
    "explain",
]

DATASET_ROWS = [
    {
        "Dataset (Profile)": "flash, qm9, ysi, pah (`reg_local_csv`)",
        "get_data": "`local_csv`",
        "curate": "`properties` (+ optional `smiles_column`, `dedupe_strategy`, `keep_all_columns`)",
        "label.normalize": "",
        "label.ic50": "",
        "split": "`holdout/cv/nested_holdout_cv` + `random/scaffold`",
        "featurize / use": "`use.curated_features`, `featurize.rdkit`, `featurize.morgan`",
        "preprocess.features": "Yes (requires feature input)",
        "select.features": "Yes (requires preprocess + feature input)",
        "train": "`random_forest`, `svm`, `decision_tree`, `xgboost`, `ensemble`, `dl_*`",
        "train.tdc": "",
        "analyze.stats": "",
        "analyze.eda": "",
        "explain": "Yes",
    },
    {
        "Dataset (Profile)": "chembl (`reg_chembl_ic50`)",
        "get_data": "`chembl`",
        "curate": "`properties: standard_value`, `keep_all_columns`",
        "label.normalize": "",
        "label.ic50": "Yes",
        "split": "`holdout/cv/nested_holdout_cv` + `random/scaffold`",
        "featurize / use": "`featurize.lipinski` + `featurize.rdkit` (rdkit input required)",
        "preprocess.features": "Yes (requires feature input)",
        "select.features": "Yes (requires preprocess + feature input)",
        "train": "`random_forest`, `svm`, `decision_tree`, `xgboost`, `ensemble`, `dl_*`",
        "train.tdc": "",
        "analyze.stats": "Yes",
        "analyze.eda": "Yes",
        "explain": "Yes",
    },
    {
        "Dataset (Profile)": "pgp, ara (`clf_local_csv`)",
        "get_data": "`local_csv`",
        "curate": "`properties`, `smiles_column`, `dedupe_strategy`, `label_column`",
        "label.normalize": "Optional (required if target not already binary)",
        "label.ic50": "",
        "split": "`holdout/cv/nested_holdout_cv` + `random/scaffold`",
        "featurize / use": "`catboost_classifier`/`dl_*`: one of `use.curated_features`, `featurize.rdkit`, `featurize.morgan`; `chemprop`: none required",
        "preprocess.features": "Yes for non-chemprop (requires feature input)",
        "select.features": "Yes for non-chemprop (requires preprocess + feature input)",
        "train": "`catboost_classifier`, `chemprop`, `dl_*`",
        "train.tdc": "",
        "analyze.stats": "",
        "analyze.eda": "",
        "explain": "Yes* (chemprop explain is skipped)",
    },
    {
        "Dataset (Profile)": "pgp_tdc_benchmark (`clf_tdc_benchmark`)",
        "get_data": "",
        "curate": "",
        "label.normalize": "",
        "label.ic50": "",
        "split": "",
        "featurize / use": "",
        "preprocess.features": "",
        "select.features": "",
        "train": "",
        "train.tdc": "`catboost_classifier`",
        "analyze.stats": "",
        "analyze.eda": "",
        "explain": "",
    },
]

COMMON_NOTES = [
    "`train` and `train.tdc` are mutually exclusive.",
    "`train.tdc` must be terminal (last node).",
    "`split` must appear before `preprocess.features`, `select.features`, `train`, and `explain`.",
    "`chemprop` cannot be paired with `preprocess.features`/`select.features` in DOE validation.",
    "Runtime warns and skips explainability for chemprop.",
    "`clf_tdc_benchmark` uses a `train.tdc`-only pipeline (no `get_data` node).",
]


def _md_row(headers: list[str], row: dict[str, str]) -> str:
    return "| " + " | ".join(str(row.get(header, "")) for header in headers) + " |"


def _write_markdown(path: Path, title: str, subtitle: str, headers: list[str], rows: list[dict[str, str]]) -> None:
    lines = [
        "<!-- Generated by docs/generate_support_table.py; do not edit manually. -->",
        f"# {title}",
        "",
        subtitle,
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append(_md_row(headers, row))

    lines.extend(["", "## Notes", ""])
    for note in COMMON_NOTES:
        lines.append(f"- {note}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _inline_md_to_html(text: str) -> str:
    escaped = html.escape(str(text))
    escaped = escaped.replace("&lt;br&gt;", "<br>")
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)


def _html_table(headers: list[str], rows: list[dict[str, str]]) -> str:
    head_html = "".join(f"<th>{_inline_md_to_html(h)}</th>" for h in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{_inline_md_to_html(row.get(h, ''))}</td>" for h in headers)
        body_rows.append(f"<tr>{cells}</tr>")
    body_html = "\n        ".join(body_rows)
    return f"""
  <div class=\"wrap\">\n    <table>\n      <thead>\n        <tr>{head_html}</tr>\n      </thead>\n      <tbody>\n        {body_html}\n      </tbody>\n    </table>\n  </div>
"""


def _write_html(path: Path, page_title: str, subtitle: str, headers: list[str], rows: list[dict[str, str]]) -> None:
    notes_html = "\n    ".join(f"<li>{_inline_md_to_html(note)}</li>" for note in COMMON_NOTES)
    table_html = _html_table(headers, rows)
    content = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(page_title)}</title>
  <style>
    :root {{
      --bg: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --line: #dbe3ee;
      --head: #f1f5f9;
      --chip: #eef2ff;
    }}
    body {{
      margin: 28px;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      color: var(--text);
      background: var(--bg);
    }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    p {{ margin: 0 0 16px; color: var(--muted); }}
    .wrap {{ border: 1px solid var(--line); border-radius: 10px; overflow: auto; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 1520px; font-size: 13px; }}
    th, td {{ border: 1px solid var(--line); padding: 10px; vertical-align: top; text-align: left; }}
    th {{ position: sticky; top: 0; z-index: 1; background: var(--head); }}
    code {{ background: var(--chip); border: 1px solid #c7d2fe; border-radius: 5px; padding: 1px 5px; }}
    ul {{ margin: 14px 0 0 18px; color: var(--muted); }}
  </style>
</head>
<body>
  <!-- Generated by docs/generate_support_table.py; do not edit manually. -->
  <h1>{html.escape(page_title)}</h1>
  <p>{html.escape(subtitle)}</p>
{table_html}
  <ul>
    {notes_html}
  </ul>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def main() -> int:
    profile_title = "Dataset x Node Support Matrix (Profile View)"
    profile_subtitle = "Profile-oriented view. These rows mirror DOE profile capabilities in utilities/doe.py."
    dataset_title = "Dataset x Node Support Matrix (Dataset View)"
    dataset_subtitle = "Dataset-oriented view. Empty cell means not supported for that dataset/profile mapping."

    _write_markdown(
        DOCS_DIR / "dataset_profile_support_matrix.md",
        profile_title,
        profile_subtitle,
        PROFILE_HEADERS,
        PROFILE_ROWS,
    )
    _write_html(
        DOCS_DIR / "dataset_profile_support_matrix.html",
        profile_title,
        profile_subtitle,
        PROFILE_HEADERS,
        PROFILE_ROWS,
    )

    _write_markdown(
        DOCS_DIR / "dataset_node_support_matrix.md",
        dataset_title,
        dataset_subtitle,
        DATASET_HEADERS,
        DATASET_ROWS,
    )
    _write_html(
        DOCS_DIR / "dataset_node_support_matrix.html",
        dataset_title,
        dataset_subtitle,
        DATASET_HEADERS,
        DATASET_ROWS,
    )

    print("Wrote:")
    print("- docs/dataset_profile_support_matrix.md")
    print("- docs/dataset_profile_support_matrix.html")
    print("- docs/dataset_node_support_matrix.md")
    print("- docs/dataset_node_support_matrix.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

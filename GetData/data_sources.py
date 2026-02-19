import logging
import os
import shutil
import subprocess
import sys
import urllib.request
from typing import Callable, Dict

HTTP_TIMEOUT_SECONDS = 60


def fetch_chembl(output_file: str, source: dict) -> int:
    target_name = source.get("target_name")
    if not target_name:
        logging.error("Missing source.target_name for data_source=chembl")
        return 1

    script_path = os.path.join("GetData", "get_ChEMBL_target_full.py")
    result = subprocess.run(
        [sys.executable, script_path, target_name, output_file],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logging.error("ChEMBL fetch failed. Please retry later or use local cached data.")
        if result.stdout:
            logging.error("ChEMBL stdout:\n%s", result.stdout.strip())
        if result.stderr:
            logging.error("ChEMBL stderr:\n%s", result.stderr.strip())
        return result.returncode
    return 0


def fetch_local_csv(output_file: str, source: dict) -> int:
    input_path = source.get("path")
    if not input_path:
        logging.error("Missing source.path for data_source=local_csv")
        return 1

    if not os.path.exists(input_path):
        logging.error(f"Local CSV not found: {input_path}")
        return 1

    shutil.copyfile(input_path, output_file)
    return 0


def fetch_http_csv(output_file: str, source: dict) -> int:
    url = source.get("url")
    if not url:
        logging.error("Missing source.url for data_source=http_csv")
        return 1

    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_SECONDS) as resp, open(
            output_file, "wb"
        ) as out:
            out.write(resp.read())
    except Exception as exc:
        logging.error(f"Failed to download CSV from {url}: {exc}")
        return 1

    return 0


def _normalize_tdc_columns(df):
    rename_map = {}
    if "Drug" in df.columns and "smiles" not in df.columns:
        rename_map["Drug"] = "smiles"
    if "SMILES" in df.columns and "smiles" not in df.columns:
        rename_map["SMILES"] = "smiles"
    if "Y" in df.columns and "label" not in df.columns:
        rename_map["Y"] = "label"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def fetch_tdc(output_file: str, source: dict) -> int:
    group = source.get("group", "ADME")
    name = source.get("name")
    if not name:
        logging.error("Missing source.name for data_source=tdc")
        return 1

    try:
        if group.upper() == "ADME":
            from tdc.single_pred import ADME

            data = ADME(name=name)
        else:
            logging.error(f"Unsupported TDC group: {group}")
            return 1
    except Exception as exc:
        logging.error(f"Failed to initialize TDC dataset ({group}:{name}): {exc}")
        return 1

    try:
        df = data.get_data()
        df = _normalize_tdc_columns(df)
        df.to_csv(output_file, index=False)
        return 0
    except Exception as exc:
        logging.error(f"Failed to fetch TDC dataset ({group}:{name}): {exc}")
        return 1


DATA_SOURCE_REGISTRY: Dict[str, Callable[[str, dict], int]] = {
    "chembl": fetch_chembl,
    "local_csv": fetch_local_csv,
    "http_csv": fetch_http_csv,
    "tdc": fetch_tdc,
}


def get_data(output_file: str, data_source: str, source: dict) -> int:
    handler = DATA_SOURCE_REGISTRY.get(data_source)
    if handler is None:
        logging.error(f"Unknown data_source: {data_source}")
        return 1
    return handler(output_file, source)

import logging
import os
import shutil
import subprocess
import sys
import urllib.request
from typing import Callable, Dict

HTTP_TIMEOUT_SECONDS = 60


def fetch_chembl(output_file: str, source: dict) -> int:
    target_name = str(source.get("target_name", "")).strip()
    target_chembl_id = str(source.get("target_chembl_id", "")).strip()
    if not target_name and not target_chembl_id:
        logging.error("Missing source.target_name or source.target_chembl_id for data_source=chembl")
        return 1

    script_path = os.path.join("GetData", "get_ChEMBL_target_full.py")
    cmd = [
        sys.executable,
        script_path,
        target_name or target_chembl_id,
        output_file,
    ]
    if target_chembl_id:
        cmd.extend(["--target-chembl-id", target_chembl_id])
    result = subprocess.run(
        cmd,
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


def fetch_local_npy(output_file: str, source: dict) -> int:
    """Load a .npy time-series and persist it as the canonical raw .npz.

    Required source keys:
        path: filesystem path to a 1-D or 2-D .npy file.
    Optional:
        time_axis: 'rows', 'cols', or 'auto' (default 'auto').
                   Selects which axis carries time when the array is 2-D.
    """
    input_path = source.get("path")
    if not input_path:
        logging.error("Missing source.path for data_source=local_npy")
        return 1
    if not os.path.exists(input_path):
        logging.error("Local .npy not found: %s", input_path)
        return 1

    time_axis = str(source.get("time_axis", "auto")).strip().lower() or "auto"

    # Lazy import keeps GetData decoupled from numpy until a TS source is used.
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    try:
        from utilities import timeseries_io
    finally:
        sys.path.pop(0)

    try:
        data = timeseries_io.load_npy_timeseries(input_path, time_axis=time_axis)
    except Exception as exc:
        logging.error("Failed to load .npy time-series from %s: %s", input_path, exc)
        return 1

    timeseries_io.save_raw_timeseries(
        output_file,
        data,
        source_meta={
            "data_source": "local_npy",
            "path": str(input_path),
            "time_axis": time_axis,
            "shape": list(data.shape),
        },
    )
    return 0


def fetch_local_ts_csv(output_file: str, source: dict) -> int:
    """Load a CSV time-series (one row per timestep) into the canonical .npz.

    Required source keys:
        path: filesystem path to the CSV.
    Optional:
        has_header: bool, default True. If False, all columns are treated as data.
        time_column: str (column name) or int (index) to drop before stacking.
    """
    input_path = source.get("path")
    if not input_path:
        logging.error("Missing source.path for data_source=local_ts_csv")
        return 1
    if not os.path.exists(input_path):
        logging.error("Local time-series CSV not found: %s", input_path)
        return 1

    has_header = source.get("has_header", True)
    if isinstance(has_header, str):
        has_header = has_header.strip().lower() in {"1", "true", "yes", "on"}
    time_column = source.get("time_column")

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    try:
        from utilities import timeseries_io
    finally:
        sys.path.pop(0)

    try:
        data = timeseries_io.load_csv_timeseries(
            input_path,
            has_header=bool(has_header),
            time_column=time_column,
        )
    except Exception as exc:
        logging.error("Failed to load CSV time-series from %s: %s", input_path, exc)
        return 1

    timeseries_io.save_raw_timeseries(
        output_file,
        data,
        source_meta={
            "data_source": "local_ts_csv",
            "path": str(input_path),
            "has_header": bool(has_header),
            "time_column": time_column,
            "shape": list(data.shape),
        },
    )
    return 0


DATA_SOURCE_REGISTRY: Dict[str, Callable[[str, dict], int]] = {
    "chembl": fetch_chembl,
    "local_csv": fetch_local_csv,
    "http_csv": fetch_http_csv,
    "tdc": fetch_tdc,
    "local_npy": fetch_local_npy,
    "local_ts_csv": fetch_local_ts_csv,
}


def get_data(output_file: str, data_source: str, source: dict) -> int:
    handler = DATA_SOURCE_REGISTRY.get(data_source)
    if handler is None:
        logging.error(f"Unknown data_source: {data_source}")
        return 1
    return handler(output_file, source)

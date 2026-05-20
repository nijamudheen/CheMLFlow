import argparse
import json
import logging
import os
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from GetData.data_sources import get_data


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        return json.load(f)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Fetch raw data from a configured source.")
    parser.add_argument("output_file", type=str, help="Output CSV file path")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file")
    parser.add_argument("--data_source", type=str, default=None, help="Data source name")
    parser.add_argument("--source", type=str, default=None, help="JSON object for source config")

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        get_data_config = config.get("get_data")
        if not get_data_config:
            logging.error("get_data section is required in config")
            return 1
        data_source = get_data_config.get("data_source")
        source = get_data_config.get("source", {})
    else:
        data_source = args.data_source
        source = json.loads(args.source) if args.source else {}

    if not data_source:
        logging.error("data_source is required")
        return 1

    return get_data(args.output_file, data_source, source)


if __name__ == "__main__":
    raise SystemExit(main())

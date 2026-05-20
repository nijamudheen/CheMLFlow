import argparse
import json
import os
import shutil
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            import yaml

            return yaml.safe_load(f)
        return json.load(f)


def remove_path(path: Path, dry_run: bool) -> None:
    if not path.exists():
        return
    if dry_run:
        print(f"[dry-run] would remove {path}")
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean generated pipeline outputs.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.chembl.yaml",
        help="Path to pipeline config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without deleting.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1

    config = load_config(str(config_path))
    base_dir_value = config.get("base_dir")
    if base_dir_value is None:
        base_dir_value = (config.get("global") or {}).get("base_dir")
    if base_dir_value is None:
        print("Config missing base_dir (expected at top-level or under global).")
        return 1
    base_dir = Path(base_dir_value)

    # Known outputs
    remove_path(base_dir, args.dry_run)
    global_cfg = config.get("global") if isinstance(config.get("global"), dict) else {}
    run_dir_value = config.get("run_dir")
    if run_dir_value is None:
        run_dir_value = global_cfg.get("run_dir")
    if run_dir_value:
        remove_path(Path(str(run_dir_value)), args.dry_run)
    remove_path(Path("runs"), args.dry_run)
    remove_path(Path("config") / "generated", args.dry_run)
    remove_path(Path("results"), args.dry_run)
    remove_path(Path("urease"), args.dry_run)

    # Remove empty data/ if base_dir lives under it and it is empty
    data_dir = Path("data")
    if data_dir.exists() and data_dir.is_dir():
        try:
            next(data_dir.iterdir())
        except StopIteration:
            remove_path(data_dir, args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

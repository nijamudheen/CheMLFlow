import os

from main import main

os.environ.setdefault("CHEMLFLOW_CONFIG", "config/config.qm9.yaml")

if __name__ == "__main__":
    raise SystemExit(main())

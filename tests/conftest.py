from __future__ import annotations

import sys
from pathlib import Path

# Ensure imports like `import main` and `from utilities import splitters` work when
# running pytest from the repo root.
CHEMLFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(CHEMLFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(CHEMLFLOW_ROOT))


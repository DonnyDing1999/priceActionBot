"""Run the SPY live paper-trading loop (Alpaca). See pab/live.py for design."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("PA_INSTRUMENT", "spy")

from pab.live import main  # noqa: E402

if __name__ == "__main__":
    main()

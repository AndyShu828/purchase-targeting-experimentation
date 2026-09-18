import sys
from pathlib import Path

# scripts/ is not a package, so tests reach it through sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

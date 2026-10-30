# -*- coding: utf-8 -*-
"""Article 2 pipeline: C2 driver fits -> cross-validation -> C1 direct suppression."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ajuste_c2, validacao_cruzada, ajuste_c1

if __name__ == "__main__":
    steps = sys.argv[1:] or ["c2", "cv", "c1"]
    if "c2" in steps:
        ajuste_c2.main()
    if "cv" in steps:
        validacao_cruzada.main()
    if "c1" in steps:
        ajuste_c1.main()

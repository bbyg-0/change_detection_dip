from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
filename = ROOT / "data/T1/eelriv_15807_20037_012_201113_PL09043020_05HHHH_CX_02.grd"

data = np.fromfile(filename, dtype="<c8")

i = 0
for x in data:
    i += 1
    if not abs(x) < 1e-6:
        print(x)

print(f"Size : {i}")

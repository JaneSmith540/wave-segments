from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
rows = []
for j, symbol in enumerate(["DEMO_A", "DEMO_B", "DEMO_C"]):
    regimes = np.repeat([.002, -.0015, 0., .003, -.002], [70, 45, 55, 65, 50])
    returns = regimes + rng.normal(0, np.where(regimes == 0, .004, .009))
    close = (80 + 15*j) * np.exp(np.cumsum(returns))
    open_ = close * np.exp(rng.normal(0, .003, len(close)))
    spread = np.abs(rng.normal(.008, .004, len(close)))
    high, low = np.maximum(open_, close)*(1+spread), np.minimum(open_, close)*(1-spread)
    volume = rng.lognormal(14+j/5, .35, len(close)) * (1+np.abs(returns)*20)
    rows += list(zip([symbol]*len(close), pd.date_range("2024-01-02", periods=len(close), freq="B"), open_, high, low, close, volume))
frame = pd.DataFrame(rows, columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
path = Path("data/demo_bars.parquet")
path.parent.mkdir(exist_ok=True)
frame.to_parquet(path, index=False)
print(path)

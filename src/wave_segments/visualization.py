"""Matplotlib visualisations for segmented OHLC data."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

UNKNOWN_COLOR = "#7f8c8d"
PALETTE = ("#2e86de", "#e67e22", "#8e44ad", "#16a085", "#c0392b")

def _plt():
    import matplotlib
    # File-producing CLI runs must not depend on a desktop Qt event loop.
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt

def _colors(labels, supplied=None):
    result = dict(supplied or {})
    for label in pd.unique(pd.Series(labels, dtype="object").fillna("UNKNOWN")):
        label = str(label)
        if label.upper() == "UNKNOWN": result[label] = UNKNOWN_COLOR
        elif label not in result: result[label] = PALETTE[len(result) % len(PALETTE)]
    return result

def _positions(bars, row):
    for a, b in (("start_idx", "end_idx"), ("start_bar", "end_bar")):
        if a in row and b in row and pd.notna(row[a]) and pd.notna(row[b]):
            return max(0, int(row[a])), min(len(bars) - 1, int(row[b]))
    time = pd.to_datetime(bars.timestamp)
    for a, b in (("start_timestamp", "end_timestamp"), ("start", "end")):
        if a in row and b in row and pd.notna(row[a]) and pd.notna(row[b]):
            hits = np.flatnonzero(((time >= pd.to_datetime(row[a])) & (time <= pd.to_datetime(row[b]))).to_numpy())
            if len(hits): return int(hits[0]), int(hits[-1])
    return None

def plot_segmented_candles(ohlcv, segments, *, label_col="label", color_map=None, title=None, ax=None):
    """Candlesticks with translucent variable-length segment spans, grey UNKNOWN."""
    required = {"timestamp", "open", "high", "low", "close"}
    if required - set(ohlcv): raise ValueError(f"ohlcv missing columns: {sorted(required-set(ohlcv))}")
    if ax is None and "symbol" in ohlcv and ohlcv["symbol"].nunique() > 1:
        plt = _plt(); symbols = list(ohlcv["symbol"].drop_duplicates())
        fig, axes = plt.subplots(len(symbols), 1, figsize=(14, max(4, 3.2 * len(symbols))), squeeze=False)
        shared_colors = _colors(segments.get(label_col, pd.Series(dtype="object")), color_map)
        for axis, symbol in zip(axes[:, 0], symbols):
            symbol_segments = segments[segments["symbol"] == symbol] if "symbol" in segments else segments
            plot_segmented_candles(
                ohlcv[ohlcv["symbol"] == symbol], symbol_segments, label_col=label_col,
                color_map=shared_colors, title=str(symbol), ax=axis,
            )
        fig.suptitle(title or "OHLC with variable-length wave segments", y=1.0)
        return axes[0, 0]
    bars = ohlcv.sort_values("timestamp").reset_index(drop=True).copy(); bars.timestamp = pd.to_datetime(bars.timestamp)
    plt = _plt()
    if ax is None: _, ax = plt.subplots(figsize=(14, 6))
    labels = segments.get(label_col, pd.Series("UNKNOWN", index=segments.index)).fillna("UNKNOWN")
    colors, seen = _colors(labels, color_map), set()
    for _, row in segments.iterrows():
        positions = _positions(bars, row)
        if positions is None: continue
        start, end = positions; label = str(row.get(label_col, "UNKNOWN"))
        ax.axvspan(start-.5, end+.5, color=colors.get(label, UNKNOWN_COLOR), alpha=.14, label=label if label not in seen else None, zorder=0); seen.add(label)
    x=np.arange(len(bars)); color=np.where(bars.close>=bars.open, "#d35400", "#2980b9")
    ax.vlines(x,bars.low,bars.high,color=color,linewidth=.8,zorder=2); height=(bars.close-bars.open).abs().clip(lower=1e-12)
    ax.bar(x,height,bottom=np.minimum(bars.open,bars.close),width=.62,color=color,edgecolor=color,linewidth=.3,zorder=3)
    step=max(1,len(bars)//10); ax.set_xticks(x[::step],bars.timestamp.dt.strftime("%Y-%m-%d").iloc[::step],rotation=35,ha="right")
    ax.set(ylabel="Price",title=title or "OHLC with variable-length wave segments"); ax.grid(axis="y",alpha=.18)
    if seen: ax.legend(title="Segment",fontsize="small")
    return ax

def plot_transition_heatmap(transitions, *, labels=None, normalize=True, ax=None):
    """Current-state → next-state heat map."""
    plt = _plt(); matrix = transitions.astype(float).copy() if isinstance(transitions,pd.DataFrame) else pd.DataFrame(np.asarray(transitions,float),index=labels,columns=labels)
    if normalize: matrix=matrix.div(matrix.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)
    if ax is None: _,ax=plt.subplots(figsize=(max(6,len(matrix.columns)*.8),max(5,len(matrix)*.7)))
    image=ax.imshow(matrix.to_numpy(),cmap="Blues",vmin=0,vmax=1 if normalize else None); ax.set_xticks(range(len(matrix.columns)),matrix.columns.astype(str),rotation=40,ha="right"); ax.set_yticks(range(len(matrix.index)),matrix.index.astype(str)); ax.set(xlabel="Next segment",ylabel="Current segment",title="Transition probability" if normalize else "Transition count")
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)): ax.text(j,i,f"{matrix.iloc[i,j]:.2f}" if normalize else f"{matrix.iloc[i,j]:.0f}",ha="center",va="center",fontsize=8)
    ax.figure.colorbar(image,ax=ax,shrink=.85); return ax

def plot_pca_scatter(features, segments=None, *, label_col="label", color_map=None, ax=None):
    """PCA scatter of numeric wave features; returns (axis, projection)."""
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    numeric=features.select_dtypes(include=np.number)
    if len(numeric)<2 or not len(numeric.columns): raise ValueError("PCA requires two rows and one numeric feature")
    xy=PCA(n_components=2 if len(numeric.columns)>1 else 1,random_state=42).fit_transform(StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(numeric)))
    out=pd.DataFrame({"pc1":xy[:,0],"pc2":xy[:,1] if xy.shape[1]>1 else 0.},index=features.index)
    out[label_col]=(segments[label_col] if segments is not None and label_col in segments else "UNLABELED"); out[label_col]=out[label_col].fillna("UNKNOWN").astype(str)
    plt=_plt()
    if ax is None: _,ax=plt.subplots(figsize=(9,6))
    colors=_colors(out[label_col],color_map)
    for label,group in out.groupby(label_col): ax.scatter(group.pc1,group.pc2,s=38,alpha=.78,color=colors[str(label)],label=str(label),edgecolors="white",linewidths=.35)
    ax.set(xlabel="PCA component 1",ylabel="PCA component 2",title="Segment feature space"); ax.legend(title="Label",fontsize="small"); ax.grid(alpha=.18)
    return ax,out

def plot_duration_distribution(segments, *, duration_col="duration", label_col="label", ax=None):
    """Log-scale duration distributions grouped by label."""
    data=segments.copy()
    if duration_col not in data:
        if {"start_idx","end_idx"}<=set(data): data[duration_col]=data.end_idx-data.start_idx+1
        else: raise ValueError(f"segments missing {duration_col!r} (and start_idx/end_idx)")
    data[label_col]=data.get(label_col,"UNKNOWN").fillna("UNKNOWN")
    groups=[(str(x),g[duration_col].dropna().to_numpy()) for x,g in data.groupby(label_col) if len(g)]
    if not groups: raise ValueError("no segment durations to plot")
    plt=_plt()
    if ax is None: _,ax=plt.subplots(figsize=(10,5))
    ax.boxplot([x[1] for x in groups],tick_labels=[x[0] for x in groups],showfliers=False); ax.set(yscale="log",ylabel="Duration (bars, log scale)",xlabel="Segment label",title="Segment duration distribution"); ax.grid(axis="y",alpha=.18)
    return ax

def plot_reliability_diagram(table, *, ax=None):
    """Plot accuracy against calibrated confidence from reliability_table output."""
    plt=_plt(); data=pd.DataFrame(table)
    if ax is None: _,ax=plt.subplots(figsize=(6,5))
    valid=data.dropna(subset=["mean_confidence","accuracy"])
    ax.plot([0,1],[0,1],"--",color="#7f8c8d",label="ideal")
    ax.plot(valid.mean_confidence,valid.accuracy,"o-",color="#2e86de",label="model")
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="Mean confidence",ylabel="Observed accuracy",title="Reliability diagram")
    ax.grid(alpha=.2); ax.legend(); return ax

def plot_coverage_risk(curve, *, ax=None):
    """Plot selective classification error as coverage changes."""
    plt=_plt(); data=pd.DataFrame(curve).dropna(subset=["coverage","risk"])
    if ax is None: _,ax=plt.subplots(figsize=(6,5))
    ax.plot(data.coverage,data.risk,color="#c0392b")
    ax.set(xlim=(0,1),ylim=(0,max(1e-6,float(data.risk.max())*1.05) if len(data) else 1),
           xlabel="Coverage",ylabel="Error among recognised segments",title="Coverage–risk curve")
    ax.grid(alpha=.2); return ax

def save_standard_visualizations(ohlcv, segments, features, transitions, output_dir, *, label_col="label"):
    """Save the four standard audit figures and return their paths."""
    plt=_plt(); output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    makers={"segmented_candles":lambda:plot_segmented_candles(ohlcv,segments,label_col=label_col).figure,"transition_heatmap":lambda:plot_transition_heatmap(transitions).figure,"pca_scatter":lambda:plot_pca_scatter(features,segments,label_col=label_col)[0].figure,"duration_distribution":lambda:plot_duration_distribution(segments,label_col=label_col).figure}
    paths={}
    for name,make in makers.items():
        fig=make(); fig.tight_layout(); path=output/f"{name}.png"; fig.savefig(path,dpi=160,bbox_inches="tight"); plt.close(fig); paths[name]=path
    return paths

def create_all_visualizations(bars, labeled_segments, transitions, out):
    """Pipeline-compatible chart writer. Numeric columns of segments feed PCA."""
    from .model import infer_feature_columns
    features=labeled_segments[infer_feature_columns(labeled_segments)].copy()
    return save_standard_visualizations(bars,labeled_segments,features,transitions,out)

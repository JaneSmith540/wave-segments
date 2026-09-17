"""Optional Streamlit UI for append-only multi-reviewer segment annotation."""
from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from hashlib import blake2b
from pathlib import Path

import pandas as pd

from .bulk_market_data import adjust_prices
from .data import load_local
from .schema import normalize_ohlcv

LABELS = ["上涨推进", "下跌推进", "高波动震荡", "低波动盘整", "反转过渡", "跳跃事件冲击", "UNKNOWN"]


def resolve_candidate_status(row: pd.Series) -> str:
    """Resolve explicit status or infer it for legacy queues without the field."""
    raw_status = row.get("candidate_status", "")
    status = "" if pd.isna(raw_status) else str(raw_status).strip()
    if status:
        return status
    if str(row.get("unknown_reason", "")) == "unreviewed_candidate":
        return "unreviewed_candidate"
    if str(row.get("predicted_label", "UNKNOWN")).upper() == "UNKNOWN":
        return "model_abstention"
    return "classified"


def load_bars_for_symbol(path: str | Path, symbol: str) -> pd.DataFrame:
    """Load one security's bars from a local file or full-market bucket dataset.

    Full-market candidate outputs keep source OHLC prices plus `adj_factor` in
    `_symbol_shards`. Load just the security's stable-hash bucket and apply the
    same per-bar HFQ convention used to build the candidate segments.
    """
    root = Path(path)
    if root.is_file():
        bars = load_local(root)
        return bars.loc[bars.symbol.astype(str).eq(str(symbol))].reset_index(drop=True)

    shard_root = root / "_symbol_shards"
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not shard_root.is_dir() or not manifest_path.is_file():
        raise ValueError(f"bars path must be an OHLCV file or full-market candidate directory: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bucket_count = int(manifest.get("bucket_count", 0))
    if bucket_count <= 0:
        raise ValueError("full-market manifest has invalid bucket_count")
    digest = blake2b(str(symbol).encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "little") % bucket_count
    bucket_dir = shard_root / f"bucket={bucket:03d}"
    files = sorted(bucket_dir.glob("shard=*.parquet"))
    if not files:
        return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
    raw = pd.read_parquet(files, filters=[("ts_code", "==", str(symbol))])
    if raw.empty:
        return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
    adjusted = adjust_prices(raw, mode="hfq")
    return normalize_ohlcv(adjusted)


def resolve_boundary_correction(value: object, symbol_bars: pd.DataFrame | None) -> tuple[int | None, str | None]:
    """Convert a reviewer-entered bar index or trading date to a symbol bar index."""
    text = "" if value is None else str(value).strip()
    if not text:
        return None, None
    try:
        number = float(text)
        if number.is_integer() and number >= 0:
            return int(number), None
    except ValueError:
        pass
    if symbol_bars is None or symbol_bars.empty or "timestamp" not in symbol_bars:
        raise ValueError("日期型边界修正需要 --bars，以便转换为交易 K 线序号")
    timestamp = pd.to_datetime(text, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError("边界修正必须是非负 K 线序号或有效日期")
    ordered = symbol_bars.sort_values("timestamp").reset_index(drop=True)
    matches = ordered.index[pd.to_datetime(ordered["timestamp"]).dt.normalize().eq(timestamp.normalize())]
    if len(matches) != 1:
        raise ValueError("修正日期不是该股票唯一的交易日，请选择实际交易日")
    return int(matches[0]), timestamp.isoformat()


def append_annotation(path: str | Path, record: dict[str, object]) -> pd.DataFrame:
    """Append one immutable review event using an atomic file replacement."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(target, dtype={"segment_id": str}) if target.exists() else pd.DataFrame()
    row = {**record}
    row.setdefault("annotation_id", str(uuid.uuid4()))
    row.setdefault("reviewed_at", datetime.now(timezone.utc).isoformat())
    updated = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    updated.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(target)
    return updated


def run_app(queue_path: str, bars_path: str | None, annotations_path: str) -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Install annotation UI: pip install -e '.[annotation]'") from exc
    queue = pd.read_csv(queue_path, dtype={"segment_id": str}).sort_values("review_priority", ascending=False)
    st.set_page_config(page_title="波段人工审核", layout="wide")
    st.title("波段人工审核（UNKNOWN 优先）")
    reviewer = st.text_input("审核者 ID")
    position = st.number_input("样本序号", 0, max(0, len(queue) - 1), 0)
    row = queue.iloc[int(position)]
    status_help = {
        "unreviewed_candidate": "待发现候选：尚未运行类别发现器，不是模型弃权。",
        "model_abstention": "模型弃权：发现器已运行，但依据 UNKNOWN 门控拒绝分类。",
        "classified": "已有临时类别：这是非语义簇标签，尚非人工确认语义。",
    }
    candidate_status = resolve_candidate_status(row)
    st.info(status_help.get(candidate_status, f"候选状态：{candidate_status}"))
    st.dataframe(row.to_frame("value"), use_container_width=True)
    symbol_bars = None
    if bars_path:
        from .visualization import plot_segmented_candles
        symbol_bars = load_bars_for_symbol(bars_path, str(row.symbol))
        if not symbol_bars.empty:
            start, end = pd.Timestamp(row.start_timestamp), pd.Timestamp(row.end_timestamp)
            pad = pd.Timedelta(days=45)
            # Never reveal bars after the segment end during shape labelling.
            view = symbol_bars[symbol_bars.timestamp.between(start - pad, end)]
            segment = pd.DataFrame([{**row.to_dict(), "start": start, "end": end, "label": row.predicted_label}])
            st.pyplot(plot_segmented_candles(view, segment).figure)
    with st.form("annotation"):
        label = st.selectbox("真实类别", LABELS)
        boundary_ok = st.selectbox("边界是否合理", ["yes", "no", "uncertain"])
        start_fix = st.text_input("建议开始边界（可填交易日期或 K 线序号）")
        end_fix = st.text_input("建议结束边界（可填交易日期或 K 线序号）")
        notes = st.text_area("备注")
        submitted = st.form_submit_button("追加审核记录")
    if submitted:
        if not reviewer.strip():
            st.error("必须填写审核者 ID")
        else:
            try:
                start_fix_idx, start_fix_time = resolve_boundary_correction(start_fix, symbol_bars)
                end_fix_idx, end_fix_time = resolve_boundary_correction(end_fix, symbol_bars)
            except ValueError as exc:
                st.error(str(exc))
                return
            append_annotation(annotations_path, {
                "segment_id": str(row.segment_id), "annotator": reviewer.strip(), "label": label,
                "candidate_status": candidate_status,
                "sampling_stratum": row.get("sampling_stratum"),
                "sampling_probability": row.get("sampling_probability"),
                # Repeat the immutable segment coordinates in the event.  They
                # let the offline agreement report compare corrected boundaries
                # without guessing a calendar-bar distance from dates.
                "symbol": row.get("symbol"), "start_timestamp": row.get("start_timestamp"),
                "end_timestamp": row.get("end_timestamp"), "start_idx": row.get("start_idx"),
                "end_idx": row.get("end_idx"),
                "boundary_ok": boundary_ok, "boundary_start_correction": start_fix_idx,
                "boundary_end_correction": end_fix_idx,
                "boundary_start_timestamp_correction": start_fix_time,
                "boundary_end_timestamp_correction": end_fix_time, "notes": notes,
            })
            st.success("审核事件已追加保存，不覆盖其他审核者记录。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the segment annotation UI")
    parser.add_argument("--queue", default="outputs/human_review.csv")
    parser.add_argument("--bars")
    parser.add_argument("--annotations", default="outputs/annotations.csv")
    args = parser.parse_args()
    run_app(args.queue, args.bars, args.annotations)


if __name__ == "__main__":
    main()

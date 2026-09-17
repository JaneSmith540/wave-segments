from __future__ import annotations

import argparse

from .config import PipelineConfig
from .data import fetch_tushare, load_local
from .pipeline import WavePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Variable-length probabilistic wave description")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Local CSV or Parquet OHLCV file")
    source.add_argument("--symbols", nargs="+", help="Tushare symbols, e.g. 000001.SZ")
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20261231")
    parser.add_argument("--freq", default="D")
    parser.add_argument("--adj", default="qfq", choices=["qfq", "hfq", "none"])
    parser.add_argument("--config", help="JSON configuration file")
    parser.add_argument("--output", help="Override output directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()
    if args.output:
        config.output_dir = args.output
    bars = load_local(args.input) if args.input else fetch_tushare(
        args.symbols, args.start, args.end, args.freq, None if args.adj == "none" else args.adj
    )
    result = WavePipeline(config).fit_run(bars)
    known = (result.labels["label"] != "UNKNOWN").mean()
    print(f"segments={len(result.segments)} coverage={known:.1%} output={config.output_dir}")


if __name__ == "__main__":
    main()

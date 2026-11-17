"""Command-line interface: `python -m dermxai <subcommand> [--config configs/resnet18.yaml]`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dermxai.config import Config, load_config
from dermxai.utils import resolve, setup_logging

logger = logging.getLogger("dermxai")

DEFAULT_CONFIG = "configs/resnet18.yaml"
NOT_YET = {"train": 1, "explain": 2, "shortcut-test": 3, "sanity": 4, "tag": 5, "report": 6}


def _cmd_download(cfg: Config, args: argparse.Namespace) -> int:
    """List (default) or download (--yes) the HAM10000 files, optionally unzip them (--extract)."""
    from dermxai.data.download import download, extract, verify_local_files

    raw = resolve(cfg.paths.raw_dir)
    if args.extract_only:
        verify_local_files(raw)  # manual-download path: same MD5 checks as the automatic one
    else:
        download(raw, cfg.runtime.min_free_disk_gb, confirm=args.yes)
    if args.extract or args.extract_only:
        counts = extract(raw, cfg.runtime.min_free_disk_gb)
        logger.info("Extracted file counts: %s", counts)
    return 0


def _cmd_verify(cfg: Config, args: argparse.Namespace) -> int:
    """Check counts, masks and metadata consistency."""
    from dermxai.data.verify import verify

    ok = verify(resolve(cfg.paths.raw_dir), resolve(cfg.paths.results_dir) / "data" / "verify_report.json")
    return 0 if ok else 1


def _cmd_preprocess(cfg: Config, args: argparse.Namespace) -> int:
    """Resize all images+masks once into the cache."""
    import cv2

    from dermxai.data.preprocess import preprocess
    from dermxai.data.verify import load_metadata
    from dermxai.utils import lower_process_priority, safe_num_threads

    cv2.setNumThreads(safe_num_threads(cfg.runtime.num_threads))  # keep one core free
    if cfg.runtime.low_priority:
        lower_process_priority()
    raw = resolve(cfg.paths.raw_dir)
    ids = sorted(load_metadata(raw)["image_id"])
    preprocess(raw, resolve(cfg.paths.cache_dir), ids, cfg.data.cache_short_side,
               cfg.runtime.min_free_disk_gb, limit=args.limit)
    return 0


def _cmd_split(cfg: Config, args: argparse.Namespace) -> int:
    """Create the lesion-level split and save data/splits.csv (refuses to change an existing one)."""
    import pandas as pd

    from dermxai.data.split import make_lesion_split, split_summary, summarise_split
    from dermxai.data.verify import load_metadata

    df = load_metadata(resolve(cfg.paths.raw_dir))
    split = make_lesion_split(df, cfg.data.val_frac, cfg.data.test_frac, cfg.data.split_seed)
    out = resolve(cfg.paths.splits_csv)
    if out.exists():
        old = pd.read_csv(out)
        if old.equals(split):
            logger.info("%s already exists and is identical; nothing to do.", out)
        elif not args.force:
            logger.error("%s exists and DIFFERS from the new split. Results trained on the old split would "
                         "be invalidated. Re-run with --force only if you really mean it.", out)
            return 1
    if not out.exists() or args.force:
        out.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(out, index=False)
        logger.info("Wrote %s", out)
    summary = split_summary(split)
    summary_path = resolve(cfg.paths.results_dir) / "data" / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Lesions per split: %s", summary["lesions_per_split"])
    logger.info("Share of images per split: %s", summary["image_share_per_split"])
    logger.info("Images per class and split:\n%s", summarise_split(split).to_string())
    logger.info("Saved %s", summary_path)
    return 0


def _not_yet(name: str) -> int:
    logger.error("'%s' is implemented in Phase %d and is not available yet.", name, NOT_YET[name])
    return 2


def build_parser() -> argparse.ArgumentParser:
    """Create the argparse parser with all subcommands."""
    p = argparse.ArgumentParser(prog="python -m dermxai", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, help_: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", default=DEFAULT_CONFIG, help=f"YAML config (default {DEFAULT_CONFIG})")
        return sp

    d = add("download", "list / download / unzip HAM10000 from Harvard Dataverse")
    d.add_argument("--yes", action="store_true", help="actually download (otherwise list sizes only)")
    d.add_argument("--extract", action="store_true", help="unzip after downloading")
    d.add_argument("--extract-only", action="store_true", help="only unzip already-downloaded files")
    add("verify-data", "check the raw dataset and write results/data/verify_report.json")
    pp = add("preprocess", "resize images+masks once into data/cache/")
    pp.add_argument("--limit", type=int, default=None, help="process only N pairs (timing run)")
    s = add("split", "lesion-level stratified train/val/test split -> data/splits.csv")
    s.add_argument("--force", action="store_true", help="overwrite an existing, different splits.csv")
    for name in NOT_YET:
        add(name, f"(Phase {NOT_YET[name]})")
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point used by `python -m dermxai`."""
    args = build_parser().parse_args(argv)
    cfg = load_config(resolve(args.config))
    log_dir = resolve(cfg.paths.results_dir) / ("data" if args.command in {"download", "verify-data", "preprocess", "split"} else cfg.run_name)
    setup_logging(log_dir / "log.txt")
    handlers = {"download": _cmd_download, "verify-data": _cmd_verify,
                "preprocess": _cmd_preprocess, "split": _cmd_split}
    if args.command in handlers:
        try:
            return handlers[args.command](cfg, args)
        except KeyboardInterrupt:
            logger.warning("Stopped by Ctrl+C. Nothing is corrupted; re-run the same command to resume.")
            return 130
        except (RuntimeError, OSError) as err:  # includes requests' HTTP errors
            logger.error("%s failed: %s", args.command, err)
            if args.command == "download":
                logger.error("If the Dataverse API refuses automated access, follow the manual steps "
                             "in README.md ('How to reproduce') and run: download --extract-only")
            return 1
    return _not_yet(args.command)


if __name__ == "__main__":
    sys.exit(main())

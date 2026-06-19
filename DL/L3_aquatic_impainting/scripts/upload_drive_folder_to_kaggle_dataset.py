#!/usr/bin/env python3
"""Download a Google Drive folder and upload it as a Kaggle dataset."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence


DRIVE_FOLDER_PATTERNS = (
    re.compile(r"/folders/([A-Za-z0-9_-]+)"),
    re.compile(r"[?&]id=([A-Za-z0-9_-]+)"),
)


def run(command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command))
    return subprocess.run(
        list(command),
        check=check,
        text=True,
        stdout=sys.stdout if check else subprocess.PIPE,
        stderr=sys.stderr if check else subprocess.PIPE,
    )


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(
            f"Missing `{name}` executable. Install/configure it first, then retry."
        )


def require_python_module(module_name: str, install_hint: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"Missing Python module `{module_name}` for {sys.executable}.\n"
            f"Install it with:\n  {install_hint}"
        )


def parse_drive_folder_id(value: str) -> str:
    for pattern in DRIVE_FOLDER_PATTERNS:
        match = pattern.search(value)
        if match:
            return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", value):
        return value
    raise SystemExit(
        "Drive folder must be a folder URL like "
        "https://drive.google.com/drive/folders/<id> or a raw folder id."
    )


def slug_to_title(dataset_id: str) -> str:
    slug = dataset_id.split("/", 1)[-1]
    return slug.replace("-", " ").replace("_", " ").title()


def validate_dataset_id(dataset_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", dataset_id):
        raise SystemExit("Kaggle dataset id must use format `owner/dataset-slug`.")


def make_staging_dir(args: argparse.Namespace) -> Path:
    if args.staging_dir:
        staging = Path(args.staging_dir).expanduser().resolve()
    else:
        safe_slug = args.dataset_id.split("/", 1)[1]
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        staging = (
            Path(args.staging_root).expanduser().resolve() / f"{safe_slug}-{stamp}"
        )

    if staging.exists() and any(staging.iterdir()) and not args.allow_existing_staging:
        raise SystemExit(
            f"Staging dir is not empty: {staging}\n"
            "Use a new path or pass `--allow-existing-staging`."
        )

    staging.mkdir(parents=True, exist_ok=True)
    return staging


def write_metadata(args: argparse.Namespace, staging: Path) -> None:
    metadata_path = staging / "dataset-metadata.json"
    if metadata_path.exists():
        print(f"Warning: overwriting Kaggle metadata file at {metadata_path}")

    metadata = {
        "title": args.title or slug_to_title(args.dataset_id),
        "id": args.dataset_id,
        "licenses": [{"name": args.license}],
        "subtitle": args.subtitle,
        "description": args.description,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def download_drive_folder(drive_folder: str, staging: Path, quiet: bool) -> None:
    command = [
        sys.executable,
        "-m",
        "gdown",
        "--folder",
        drive_folder,
        "-O",
        str(staging),
        "--remaining-ok",
    ]
    if quiet:
        command.append("--quiet")
    run(command)


def dataset_exists(dataset_id: str) -> bool:
    result = run(["kaggle", "datasets", "files", dataset_id], check=False)
    return result.returncode == 0


def upload_dataset(args: argparse.Namespace, staging: Path) -> None:
    mode = args.mode
    if mode == "auto":
        mode = "version" if dataset_exists(args.dataset_id) else "create"

    if mode == "create":
        command = ["kaggle", "datasets", "create", "-p", str(staging), "--dir-mode", args.dir_mode]
        if args.public:
            command.append("--public")
    else:
        command = [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(staging),
            "-m",
            args.version_message,
            "--dir-mode",
            args.dir_mode,
        ]

    if args.quiet:
        command.append("--quiet")
    run(command)


def count_payload_files(staging: Path) -> int:
    return sum(
        1
        for path in staging.rglob("*")
        if path.is_file() and path.name != "dataset-metadata.json"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a Google Drive folder and upload it to Kaggle as a dataset."
    )
    parser.add_argument("drive_folder", help="Google Drive folder URL or raw folder id.")
    parser.add_argument("dataset_id", help="Kaggle dataset id: owner/dataset-slug.")
    parser.add_argument("--title", help="Dataset title. Defaults from dataset slug.")
    parser.add_argument(
        "--subtitle",
        default="Imported from a Google Drive folder.",
        help="Dataset subtitle.",
    )
    parser.add_argument(
        "--description",
        default="Dataset uploaded from a Google Drive folder via Kaggle CLI.",
        help="Dataset description.",
    )
    parser.add_argument("--license", default="CC0-1.0", help="Kaggle license name.")
    parser.add_argument(
        "--mode",
        choices=("auto", "create", "version"),
        default="auto",
        help="Upload mode. auto creates if missing, versions if dataset exists.",
    )
    parser.add_argument(
        "--version-message",
        default="Upload Google Drive folder contents",
        help="Message for `kaggle datasets version`.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create public dataset. Default create mode is private.",
    )
    parser.add_argument(
        "--dir-mode",
        choices=("skip", "zip", "tar"),
        default="zip",
        help="Kaggle CLI handling for nested directories.",
    )
    parser.add_argument(
        "--staging-root",
        default="kaggle/drive_dataset_uploads",
        help="Root folder for timestamped staging dirs.",
    )
    parser.add_argument("--staging-dir", help="Exact staging folder to use.")
    parser.add_argument(
        "--allow-existing-staging",
        action="store_true",
        help="Allow using a non-empty staging folder. No files are deleted.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use existing staging content and only write metadata/upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and write metadata, but do not upload to Kaggle.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce CLI output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_dataset_id(args.dataset_id)
    parse_drive_folder_id(args.drive_folder)

    require_executable("kaggle")
    run(["kaggle", "--version"])
    require_python_module(
        "gdown",
        f"{sys.executable} -m pip install gdown",
    )

    staging = make_staging_dir(args)
    if not args.skip_download:
        download_drive_folder(args.drive_folder, staging, args.quiet)
    write_metadata(args, staging)

    payload_count = count_payload_files(staging)
    print(f"Staging: {staging}")
    print(f"Payload files: {payload_count}")
    if payload_count == 0:
        raise SystemExit("No payload files found. Check Drive folder access.")

    if args.dry_run:
        print("Dry run complete. Upload skipped.")
        return 0

    upload_dataset(args, staging)
    print(f"Done: {args.dataset_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create a Kaggle script job that mirrors a Google Drive folder to a Kaggle dataset.

This generator does not download the Drive folder locally. It writes a small Kaggle
kernel/script under `kaggle/drive_to_dataset/input/`; that script performs the
download and dataset upload inside Kaggle.
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Sequence


def validate_slug(value: str, label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", value):
        raise SystemExit(f"{label} must use format `owner/slug`.")


def title_from_slug(dataset_id: str) -> str:
    return dataset_id.split("/", 1)[1].replace("-", " ").replace("_", " ").title()


def build_main_py(args: argparse.Namespace) -> str:
    title = args.title or title_from_slug(args.dataset_id)
    return f'''\
#!/usr/bin/env python3
"""Run inside Kaggle: download Drive folder, upload/version Kaggle dataset."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DRIVE_FOLDER = {args.drive_folder!r}
DATASET_ID = {args.dataset_id!r}
DATASET_TITLE = {title!r}
DATASET_SUBTITLE = {args.subtitle!r}
DATASET_DESCRIPTION = {args.description!r}
DATASET_LICENSE = {args.license!r}
UPLOAD_MODE = {args.mode!r}
DIR_MODE = {args.dir_mode!r}
PUBLIC = {args.public!r}
VERSION_MESSAGE = {args.version_message!r}
STAGING = Path("/kaggle/working/drive_dataset_staging")


def run(command, check=True):
    print("+ " + " ".join(map(str, command)))
    return subprocess.run(command, check=check, text=True)


def ensure_package(package):
    try:
        __import__(package)
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", package])


def secret(name):
    value = os.environ.get(name)
    if value:
        return value
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def configure_kaggle_auth():
    username = secret("KAGGLE_USERNAME")
    key = secret("KAGGLE_KEY") or secret("KAGGLE_API_TOKEN")
    if not username or not key:
        raise SystemExit(
            "Missing Kaggle credentials. Add Kaggle notebook secrets "
            "`KAGGLE_USERNAME` and `KAGGLE_KEY` (or `KAGGLE_API_TOKEN`), "
            "or set env vars."
        )
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key


def write_metadata():
    metadata = {{
        "title": DATASET_TITLE,
        "id": DATASET_ID,
        "licenses": [{{"name": DATASET_LICENSE}}],
        "subtitle": DATASET_SUBTITLE,
        "description": DATASET_DESCRIPTION,
    }}
    (STAGING / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\\n",
        encoding="utf-8",
    )


def payload_count():
    return sum(
        1 for path in STAGING.rglob("*")
        if path.is_file() and path.name != "dataset-metadata.json"
    )


def dataset_exists():
    result = subprocess.run(
        ["kaggle", "datasets", "files", DATASET_ID],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def upload_dataset():
    mode = UPLOAD_MODE
    if mode == "auto":
        mode = "version" if dataset_exists() else "create"

    if mode == "create":
        command = ["kaggle", "datasets", "create", "-p", str(STAGING), "--dir-mode", DIR_MODE]
        if PUBLIC:
            command.append("--public")
    else:
        command = [
            "kaggle", "datasets", "version",
            "-p", str(STAGING),
            "-m", VERSION_MESSAGE,
            "--dir-mode", DIR_MODE,
        ]
    run(command)


def main():
    ensure_package("gdown")
    configure_kaggle_auth()

    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True, exist_ok=True)

    run(["gdown", "--folder", DRIVE_FOLDER, "-O", str(STAGING), "--remaining-ok"])
    write_metadata()

    count = payload_count()
    print(f"Payload files: {{count}}")
    if count == 0:
        raise SystemExit("No files downloaded. Check Drive folder sharing/access.")

    upload_dataset()
    print(f"Done: {{DATASET_ID}}")


if __name__ == "__main__":
    main()
'''


def build_kernel_metadata(args: argparse.Namespace) -> dict[str, object]:
    return {
        "id": args.kernel_id,
        "title": args.kernel_title,
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a Kaggle job that uploads a Drive folder as a Kaggle dataset."
    )
    parser.add_argument("drive_folder", help="Google Drive folder URL or id.")
    parser.add_argument("dataset_id", help="Target Kaggle dataset id: owner/dataset-slug.")
    parser.add_argument("kernel_id", help="Kaggle kernel id to create/update: owner/kernel-slug.")
    parser.add_argument("--kernel-title", default="Drive to Kaggle Dataset Uploader")
    parser.add_argument("--title", help="Dataset title. Defaults from dataset slug.")
    parser.add_argument("--subtitle", default="Imported from a Google Drive folder.")
    parser.add_argument(
        "--description",
        default="Dataset uploaded from a Google Drive folder by a Kaggle script job.",
    )
    parser.add_argument("--license", default="CC0-1.0")
    parser.add_argument("--mode", choices=("auto", "create", "version"), default="auto")
    parser.add_argument("--version-message", default="Upload Google Drive folder contents")
    parser.add_argument("--dir-mode", choices=("skip", "zip", "tar"), default="zip")
    parser.add_argument("--public", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="kaggle/drive_to_dataset/input",
        help="Local folder for generated Kaggle job files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_slug(args.dataset_id, "dataset_id")
    validate_slug(args.kernel_id, "kernel_id")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "main.py").write_text(
        textwrap.dedent(build_main_py(args)),
        encoding="utf-8",
    )
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(build_kernel_metadata(args), indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote Kaggle job to: {output_dir}")
    print("Push with:")
    print(f"  kaggle kernels push -p {output_dir}")
    print("Before run: add Kaggle notebook secrets KAGGLE_USERNAME and KAGGLE_KEY.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

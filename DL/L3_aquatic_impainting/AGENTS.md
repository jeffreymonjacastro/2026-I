# AGENTS.md

## Project

- Kaggle CS5364 underwater inpainting: reconstruct only hidden 32x32 patches in 256x256 underwater test images, then submit discrete visual IDs.

## Layout

- `AGENTS.md`: repo guidance for future coding agents.
- `.agents/skills/agents-md-compact/`: local skill used to keep this file compact.
- `kaggle/`: Kaggle jobs, notebooks, datasets, or competition artifacts when present.
- `scripts/`: local helper scripts when present.

## Competition Rules

- Image grid is 8x8 patches, row-major indices `0` to `63`, from left to right, from top to bottom; each patch is 32x32 pixels, like this:

```
 0  1  2  3  4  5  6  7
 8  9 10 11 12 13 14 15
16 17 18 19 20 21 22 23
24 25 26 27 28 29 30 31
32 33 34 35 36 37 38 39
40 41 42 43 44 45 46 47
48 49 50 51 52 53 54 55
56 57 58 59 60 61 62 63
```

- Training images are complete. Test images have 8 to 20 blacked-out patches.
- `target.csv` is source of truth for which `{image_id}_{patch_index}` rows need predictions.
- Preserve all visible test-image pixels. Do not denoise, resize, recolor, normalize-save, or otherwise rewrite unmasked regions.
- Reconstruct only hidden patches listed in `target.csv`.
- Final `submission.csv` columns must be exactly `Id,Target`; `Target` is integer codebook ID `0..1023`.
- Generate discrete targets through the competition `dino_vq.py` path when available; do not compare or optimize submission by raw pixel metrics alone.

## Commands

- Inspect repo files, including hidden/project files: `rg --files -uu`
- Check worktree before edits: `git status --short`
- Push an existing Kaggle script job only if `kaggle/drive_to_dataset/input/kernel-metadata.json` and `main.py` exist: `kaggle kernels push -p kaggle\drive_to_dataset\input`
- Run Deep Learning jobs on Kaggle GPU from `kaggle\<vn>\input`: `kaggle kernels push -p . --accelerator NvidiaTeslaT4`
- Poll long Kaggle run status: `kaggle kernels status <owner/kernel-slug>`
- If working with notebooks, validate JSON before push: `python -m json.tool path\to\main.ipynb > NUL`
- Run final DINO/VQ validation: `python dino_vq.py --images-root generate --target-csv target.csv --codebook-path codebook.npy --output-csv submission_<vn>.csv --batch-size 8`
- Submit final CSV for version `vn`: `kaggle competitions submit 2026-i-aquatic-inpainting -f submission_<vn>.csv -m "vn"`
- Check submission scores: `kaggle competitions submissions 2026-i-aquatic-inpainting --csv`

## Working Rules

- For any Kaggle CLI/API work, use the global Kaggle skill at `C:\Users\jeffr\.agents\skills\kaggle\SKILL.md`; for remote GPU jobs, read its `references/runner.md`.
- For Deep Learning experiments, prefer Kaggle remote jobs with GPU over local training; keep one version per `kaggle/<vn>/input/` and download outputs to `kaggle/<vn>/outputs/`.
- Kaggle GPU executions usually take more than 1 hour; launch them in a background shell/session, keep the thread free to poll `kaggle kernels status <owner/kernel-slug>`, and wait for `COMPLETE` before downloading outputs.
- Keep `enable_gpu: true` in `kernel-metadata.json` when using `--accelerator NvidiaTeslaT4`.
- Kaggle kernel metadata must include the competition dataset input: `"dataset_sources": ["jeffreyamc/lab3-dl-aquatic-impainting-dataset"]`. This dataset contains `train/`, `test/`, `target.csv`, `dino_vq.py`, and `codebook.npy`.
- Keep Kaggle credentials out of code, notebooks, metadata, and commits. Use Kaggle Secrets or environment variables.
- For DINOv3/Hugging Face access in notebooks, use environment variables to access the temporal token.
- First `kaggle kernels push` for a new kernel may run and fail until the user manually creates/enables Kaggle Secrets `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN`; repeated pushes to the same `kernel-metadata.json` `id` should preserve existing secrets, but new kernel IDs need manual secret setup again.
- If reusing a Kaggle notebook/kernel, preserve its real slug and secrets context first; verify with Kaggle UI or `kaggle kernels list --mine --search <term>` before changing `kernel-metadata.json`.
- If a Kaggle push fails with `Your kernel title does not resolve to the specified id` or `409 Client Error: Conflict`, check `kernel-metadata.json` `id` against the real Kaggle slug before editing other files.
- If runtime says `Missing Kaggle credentials`, fix Kaggle Secrets/env names; `kaggle kernels push` cannot attach secrets.
- For Kaggle notebooks, use readable Markdown sections before code chunks, stable cell IDs, and small persisted artifacts such as `run_summary.json`; split into sections such as overview, imports, variables/config, paths, data validation, model/inference, submission writing, and summary.
- If notebook work needs temporary helper scripts, create them under `scripts/` and call/import them from the notebook.
- Do not run Kaggle CLI or Kaggle API submission commands inside the notebook. The notebook should write `submission_<vn>.csv` as an output; Codex should download it, run `kaggle competitions submit ...`, then call `kaggle competitions submissions 2026-i-aquatic-inpainting --csv` outside the notebook to read the score.
- Prefer compact top-level ZIP outputs over many loose files when downloading Kaggle outputs on Windows.
- For Kaggle downloads in this Windows terminal, use `python -X utf8 -m kaggle ...` instead of `kaggle ...` when long filenames or non-ASCII paths may trigger `charmap` encoding errors.
- After producing `submission_<vn>.csv`, submit it with message equal to the code version (`v1`, `v2`, etc.), read the matching row from `kaggle competitions submissions 2026-i-aquatic-inpainting --csv`, and iterate if `status` is not `SubmissionStatus.COMPLETE` or `publicScore < 0.5`.

## Avoid

- Do not modify visible regions of test images.
- Do not submit one row per image; submit exactly one row per hidden patch in `target.csv`.
- Do not assume dataset mounts are always `/kaggle/input/<slug>`; check nested Kaggle mount paths when code runs remotely.
- Do not put all notebook logic in one code cell or skip Markdown section headers.
- Do not execute `kaggle competitions submit`, `kaggle competitions submissions`, or Kaggle API submission calls from inside Kaggle notebooks.
- Do not recreate deleted tracked helper files unless the user asks for that workflow.
- Do not add frameworks or new config until an actual script/notebook needs them.
- Do not churn kernel IDs between versions unless needed; keeping the same Kaggle kernel preserves manually configured secrets.

# Module A — Speaker Recognition Research Workspace

Module A runs separately from the FastAPI/React application. It prepares, trains,
evaluates, calibrates, and eventually exports the speaker model consumed by Track B.
The current implementation stops at A1 and contains no model loading or training code.

## Milestones

- A0: structure, lightweight configuration, reproducibility, and stable contracts — implemented.
- A1: dataset discovery, audio metadata, eligibility filtering, speaker-disjoint manifests — implemented.
- A2: model forward/backward sanity — not implemented.
- A3: training and checkpointing — not implemented.
- A4: SV/SID evaluation and validation-only calibration — not implemented.
- A5: export and local Track B ABI smoke — not implemented.

## Environment

Python 3.11+ is required. WAV inspection uses only the standard library. FLAC, MP3,
and M4A header inspection uses `ffprobe`, normally installed with ffmpeg.

```bash
python -m venv .venv-module-a
.venv-module-a/Scripts/python -m pip install -r module_a/requirements.txt
```

On Linux/Kaggle use `.venv-module-a/bin/python` if creating a virtual environment.
Kaggle images usually already include NumPy, PyYAML, pytest, and ffmpeg; no model is
downloaded by A0/A1.

## Configuration

`configs/dataset.yaml` is the A1 source of truth. The dataset root is intentionally
`null` and should be supplied at runtime:

```bash
python -m module_a.scripts.inspect_dataset --dataset-root /kaggle/input/vietnam-celeb
python -m module_a.scripts.prepare_manifests --dataset-root /kaggle/input/vietnam-celeb
```

Use `--output-dir /kaggle/working/module-a-outputs` to override the output directory.
`inspection.max_files` can select a deterministic sorted subset for layout inspection.

### Speaker ID strategies

The default `speaker_id_source: parent_dir` is an explicit assumption that the
immediate audio parent is the speaker ID. The CLI prints the selected strategy. Do
not use it blindly before inspecting the real Vietnam-Celeb layout.

- `parent_dir`: immediate parent directory.
- `path_component`: component index relative to dataset root; configure
  `speaker_id_path_component` (negative indices are allowed).
- `metadata_csv`: configure `speaker_metadata_csv` and its path/ID column names.
  A relative metadata CSV path is resolved from the dataset YAML file location;
  audio paths inside the CSV are resolved relative to the dataset root.

Audio directly under the dataset root, missing metadata rows, invalid components,
or empty speaker IDs fail with a controlled error instead of receiving a guessed ID.

## A1 output contracts

Runtime files go to `module_a/outputs/` and are ignored by Git except `.gitkeep`:

- `dataset_summary.json`
- `train_manifest.csv`
- `val_manifest.csv`
- `test_manifest.csv`
- `split_summary.json`

Manifest columns are:

```text
path,speaker_id,split,duration_sec,sample_rate,channels
```

The first three columns are stable. `path` is POSIX-style and relative to the dataset
root; consumers must resolve it against the same `--dataset-root`. This avoids baking
Kaggle or local Windows absolute paths into reusable manifests.

Unreadable/corrupt audio is counted in the dataset summary and never enters a manifest.
Speakers with fewer than `min_utterances_per_speaker: 2` usable utterances are reported
and excluded. The value 2 is an engineering eligibility default, not a calibrated
quality threshold.

## Split and reproducibility policy

Speakers are sorted, shuffled with a local Python RNG using seed 42, and assigned by
speaker—not by utterance—to train/validation/test at 80%/10%/10%. Allocation uses
largest remainders, assigns every speaker exactly once, and deterministically moves a
speaker from the largest split if rounding leaves a split empty. Fewer than three
eligible speakers fail because a non-empty three-way speaker-disjoint split is
impossible.

The validator rejects duplicate paths, speaker leakage, missing files, corrupt markers,
invalid split names, empty fields, empty splits, and incomplete expected coverage.
`seed_everything(42)` seeds Python and NumPy without requiring PyTorch or a GPU in A1.

## Local verification

```bash
python -m compileall module_a
pytest module_a/tests -q
python -m module_a.scripts.inspect_dataset --help
python -m module_a.scripts.prepare_manifests --help
python -m module_a.scripts.sanity_check
```

The sanity command creates temporary 16 kHz WAV fixtures, including one corrupt file,
then proves discovery, metadata handling, 8/1/1 speaker allocation, manifest validation,
and byte-identical repeated output. It deletes the temporary dataset afterward.

## Future Track B handoff

A5 must retain this application ABI:

```python
load_model(model_dir, device="auto")
extract_embedding(model, audio_path)
```

The final embedding will be a finite, fixed-dimension, L2-normalized 1-D `np.float32`
array. A0/A1 does not modify `backend/app/models/speaker_model.py`.

Future exported fields remain `embedding_dimension`, `sv_threshold`, and
`sid_threshold`. Enrollment-quality fields remain `min_duration`, `max_duration`,
`min_speech_ratio`, `min_snr_db`, `max_clipping_ratio`, and `max_content_wer`.

Quality calibration is not part of A1. Later Module A experiments must use the exact
M1 definition: mono 16 kHz PCM16, WebRTC VAD mode 2, complete 30 ms frames, speech
ratio from the same VAD classifications, and VAD-grouped SNR using the same power,
edge-case, and clamping policy. Do not introduce a second SNR estimator.

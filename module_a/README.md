# Module A — ECAPA-TDNN on VoxVietnam

Module A is the Kaggle research workspace for one speaker encoder:

```text
16 kHz mono waveform
→ 80-bin log-Mel filterbank
→ per-utterance mean normalization
→ ECAPA-TDNN
→ 192-D L2-normalized embedding
```

The same encoder supports speaker verification (SV) and open-set speaker
identification (SID). AAM-Softmax is training-only and is not exported.

## Two-phase workflow

Phase 1 uses only VoxVietnam-T. It creates a deterministic 90/10 speaker-disjoint
split with seed 42, trains ECAPA using speaker-balanced batches, selects `best.pt`
by validation SV EER, and calibrates both deployment thresholds on validation.

```bash
python -m module_a.scripts.train_ecapa \
  --dataset-root /path/to/VoxVietnam-or-VoxVietnam-T \
  --output-dir /path/to/outputs \
  --device auto
```

Phase 2 first validates the persisted calibration against `best.pt`, then and only
then opens VoxVietnam-O. It never recalibrates on final test data.

```bash
python -m module_a.scripts.evaluate_export \
  --dataset-root /path/to/VoxVietnam-or-VoxVietnam-O \
  --output-dir /path/to/outputs \
  --device auto \
  --sv-protocol auto
```

If `--dataset-root` is omitted, both scripts use the configured Hugging Face source
`hustep-lab/VoxVietnam-Dataset`. Put a gated-dataset token in `HF_TOKEN`; do not put
tokens in YAML or source control. Phase 1 restricts the snapshot patterns to
VoxVietnam-T, while Phase 2 requests VoxVietnam-O.

`--sv-protocol auto` uses one unambiguous trial file found under VoxVietnam-O. If no
trial list is present it runs a clearly labeled custom balanced protocol. Use
`--sv-protocol official --official-trials /path/to/trials.txt` to require a supplied
official protocol. The official file schema/path coverage must be verified on Kaggle;
custom results must not be presented as comparable to the published benchmark.

## Configuration and outputs

All defaults are in `configs/ecapa_voxvietnam.yaml`: feature dimensions, ECAPA size,
AAM margin/scale, AdamW settings, augmentation, sampling, and calibration targets.
Noise and reverb are enabled only when their configured resource directories contain
usable WAV/FLAC files. `training_summary.json` records what was actually enabled.

Expected output:

```text
outputs/
├── manifests/train.csv
├── manifests/validation.csv
├── checkpoints/last.pt
├── checkpoints/best.pt
├── calibration/sv_calibration.json
├── calibration/sid_calibration.json
├── metrics/validation_metrics.json
├── metrics/sv_test_metrics.json
├── metrics/sid_test_metrics.json
├── training_summary.json
├── evaluation_summary.json
└── module_a_export/
    ├── model.pt
    ├── config.json
    ├── thresholds.json
    └── metadata.json
```

Stable runtime ABI:

```python
from module_a.src.runtime import load_model, extract_embedding

model = load_model("outputs/module_a_export", device="auto")
embedding = extract_embedding(model, "/path/to/audio.wav")
```

The returned value is finite `numpy.float32`, shape `(192,)`, and L2-normalized.

## Kaggle Run All — five cells

### Cell 1 — clone/update and install

```python
import os, pathlib, subprocess

REPO_URL = "https://github.com/hvpng/Secure-Virtual-Assistant-with-Speaker-Recognition.git"
PROJECT_DIR = pathlib.Path("/kaggle/working/secure-voice-assistant")
if PROJECT_DIR.exists():
    subprocess.run(["git", "-C", str(PROJECT_DIR), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", REPO_URL, str(PROJECT_DIR)], check=True)
subprocess.run([
    "python", "-m", "pip", "install", "-q", "-r",
    str(PROJECT_DIR / "module_a" / "requirements.txt"),
], check=True)
os.chdir(PROJECT_DIR)
```

### Cell 2 — paths and dataset source

```python
import os, pathlib

PROJECT_DIR = pathlib.Path("/kaggle/working/secure-voice-assistant")
OUTPUT_DIR = pathlib.Path("/kaggle/working/voxvietnam_ecapa_outputs")

# Local/Kaggle mount: set this to the parent containing VoxVietnam-T/O, or leave empty.
DATASET_ROOT = os.environ.get("VOXVIETNAM_ROOT", "")

# Hugging Face: add HF_TOKEN through Kaggle Secrets, never inline a real token here.
HF_REPO_ID = "hustep-lab/VoxVietnam-Dataset"
if not os.environ.get("HF_TOKEN"):
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        os.environ["HF_TOKEN"] = ""  # Public/local sources may not require a token.
print({"project": str(PROJECT_DIR), "output": str(OUTPUT_DIR), "local_dataset": DATASET_ROOT or None})
```

### Cell 3 — Phase 1 train + validation calibration

```python
import subprocess

command = [
    "python", "-m", "module_a.scripts.train_ecapa",
    "--output-dir", str(OUTPUT_DIR),
    "--device", "cuda",
]
command += ["--dataset-root", DATASET_ROOT] if DATASET_ROOT else ["--hf-repo-id", HF_REPO_ID]
subprocess.run(command, cwd=PROJECT_DIR, check=True)
```

### Cell 4 — Phase 2 final evaluation + export

```python
import subprocess

command = [
    "python", "-m", "module_a.scripts.evaluate_export",
    "--output-dir", str(OUTPUT_DIR),
    "--device", "cuda",
    "--sv-protocol", "auto",
]
command += ["--dataset-root", DATASET_ROOT] if DATASET_ROOT else ["--hf-repo-id", HF_REPO_ID]
subprocess.run(command, cwd=PROJECT_DIR, check=True)
```

### Cell 5 — compact results and export listing

```python
import json

for name in ("training_summary.json", "evaluation_summary.json"):
    path = OUTPUT_DIR / name
    print(f"\n{name}")
    print(json.dumps(json.loads(path.read_text()), indent=2, ensure_ascii=False))
print("\nExported files:")
for path in sorted((OUTPUT_DIR / "module_a_export").iterdir()):
    print(path.name, path.stat().st_size)
```

## Local verification

```bash
python -m compileall module_a
pytest module_a/tests -q
python -m module_a.scripts.train_ecapa --help
python -m module_a.scripts.evaluate_export --help
```

Local tests use synthetic audio and small ECAPA channel counts. They do not access
VoxVietnam, Hugging Face, or start a full training job.

## Dataset assumptions requiring Kaggle verification

- Audio is expected below speaker directories; the default speaker ID is the second
  path component from the end (`speaker/file.wav`). Change only the YAML component
  depth if the official mounted layout includes an extra session directory.
- Hugging Face snapshot paths are expected to contain directories named
  `VoxVietnam-T` and `VoxVietnam-O`.
- Official VoxVietnam-O trial-list filename, row schema, and path prefixes must be
  checked against the actual gated distribution before claiming official metrics.

# Module A — Speaker Recognition Research Workspace

Module A runs separately from the FastAPI/React application. It prepares, trains,
evaluates, calibrates, and eventually exports the speaker model consumed by Track B.
The current implementation stops at A4. It contains Stage-1 training infrastructure
and reproducible SV/SID evaluation with validation-only calibration. Real A4 metric
values are intentionally not claimed until the frozen Kaggle checkpoint is evaluated.

## Milestones

- A0: structure, lightweight configuration, reproducibility, and stable contracts — implemented.
- A1: dataset discovery, audio metadata, eligibility filtering, speaker-disjoint manifests — implemented.
- A2: WavLM/CAM++/AAM forward, backward, optimizer, and checkpoint sanity — implemented.
- A3: train-manifest dataset, balanced mini-training, monitoring, and resume — implemented.
- A4: SV/SID evaluation and validation-only calibration — implemented.
- A5: export and local Track B ABI smoke — not implemented.

## Environment

Python 3.11+ is required. WAV inspection uses only the standard library. FLAC, MP3,
and M4A header inspection uses `ffprobe`, normally installed with ffmpeg.

```bash
python -m venv .venv-module-a
.venv-module-a/Scripts/python -m pip install -r module_a/requirements.txt
```

On Linux/Kaggle use `.venv-module-a/bin/python` if creating a virtual environment.
Kaggle images usually already include PyTorch and ffmpeg. A0/A1 and default tests do
not download a model. Only the explicit real A2 smoke command may download WavLM.

## A2 architecture sanity

The frozen Stage-1 graph is:

```text
float32 waveform [B, samples] at 16 kHz
  -> microsoft/wavlm-base-plus final hidden state [B, frames, 768]
  -> LayerNorm(768)
  -> team engineering adapter Linear(768, 80) [B, frames, 80]
  -> explicit transpose [B, 80, frames]
  -> CAM++
  -> raw speaker embedding [B, 192]
  -> AAM-Softmax head during training
```

Inference uses L2-normalized embeddings. AAM-Softmax normalizes embeddings and class
weights internally, then applies margin `0.2` and scale `30.0`. `num_classes` is a
factory argument; A2 only uses synthetic labels. A3 must derive its mapping from train
speakers only.

All WavLM parameters are frozen and the frontend remains in eval mode even while the
trainable adapter/CAM++/AAM path is in training mode. `stage2_enabled` is present only
as an explicit future gate and must remain false in A2.

Waveform collation uses deterministic center cropping and repeat-padding to the
configured 3-second segment. Repeat-padding ensures CAM++ statistics pooling does not
silently include zero padding. PCM16 WAV loading is implemented without an optional
torchaudio decoder backend; other formats use torchaudio when its runtime backend is
available.

### CAM++ source and adaptation status

The implementation is a local, non-verbatim reimplementation informed by:

- Wang et al., [CAM++: A Fast and Efficient Network for Speaker Verification Using
  Context-Aware Masking](https://arxiv.org/abs/2303.00332), Interspeech 2023.
- The Apache-2.0 [ModelScope/3D-Speaker CAM++
  implementation](https://github.com/modelscope/3D-Speaker/tree/main/speakerlab/models/campplus).

It retains the paper's frequency convolution module, densely connected TDNN backbone,
per-layer context-aware masks, global plus fixed-segment context, transition layers,
and statistics pooling. The integration differs from the original Fbank recipe because
the team's explicit WavLM adapter supplies 80-D frame features. The implementation is
not claimed to be an official CAM++ release or an exact reproduction of published
performance.

Within each dense layer, the second `BatchNorm1d -> ReLU` output is the shared input
to both the local TDNN convolution and the context-aware mask, matching the referenced
CAM++ data flow. Checkpoints trained before this correction fed the unnormalized
bottleneck tensor to the sigmoid mask; this could create near-constant dense growth
channels and degenerate downstream bottleneck running variances. Such checkpoints
must not be used for A4 calibration/export and require retraining from initialization.
`python -m module_a.scripts.diagnose_campp_checkpoint --help` documents the read-only
BN-state and known-utterance diagnostic command.

Production-size config uses CAM++ block depths `12/24/16`, growth rate 32, and a 192-D
embedding. Offline tests and `sanity_model` inject a smaller topology while preserving
all tensor contracts so CPU tests stay fast. The real smoke uses production-size config.

### A2 commands

```bash
pytest module_a/tests -q
python -m module_a.scripts.sanity_model
python -m module_a.scripts.smoke_model --help
python -m module_a.scripts.smoke_model --device auto
```

`sanity_model` injects a deterministic fake WavLM and never downloads weights. It runs
one forward/backward/optimizer step and a temporary checkpoint roundtrip. `smoke_model`
is the separate real Hugging Face integration check; it defaults to one short waveform,
batch size one, frozen WavLM, and no optimizer.

The atomic checkpoint contract stores `model_state_dict`, `optimizer_state_dict`,
`epoch`, `step`, the full serialized config, `num_classes`, and the train-speaker-only
`speaker_to_index` mapping. It stores model tensors, not a Hugging Face cache directory.

A2 proves model-graph correctness only. A3 adds bounded training infrastructure but
does not claim a completed Vietnam-Celeb training result. EER, SID evaluation, and all
thresholds belong to A4; production Track B export belongs to A5.

## A3 mini-training

The AAM classifier is built exclusively from sorted speaker IDs in the A1 **train**
manifest. A1 validation and test speakers never enter `speaker_to_index` and are not
used for AAM classification loss. They remain reserved for A4 embedding-based SV/SID
evaluation and validation-only calibration.

A3 creates a deterministic internal monitor holdout from train-manifest utterances:

- selected train speakers define all classifier classes;
- monitor speakers are a deterministic subset of those same classifier speakers;
- for each monitor speaker, `floor(utterances * monitor_holdout_ratio)` samples are
  held out, with a minimum of one monitor utterance and at least one fit utterance left;
- one-utterance speakers remain in fit data and cannot enter the monitor subset;
- fit and monitor paths are disjoint;
- monitor audio uses deterministic center crop/repeat-pad, while fit audio uses seeded
  random crop and the same repeat-padding policy for short files.

Training uses a speaker-balanced batch sampler. Each batch first chooses
`speakers_per_batch` distinct speakers uniformly, then chooses
`utterances_per_speaker` samples uniformly for each speaker. A low-resource speaker is
sampled with replacement only when it has too few fit utterances. The sequence is
deterministic for the same seed and sampler epoch, so high-resource speakers do not
dominate merely because they have more files.

The Stage-1 optimizer remains AdamW over trainable adapter/CAM++/AAM parameters only;
WavLM is frozen and excluded. The scheduler is per-update warmup plus cosine decay.
CUDA uses `torch.autocast` and `torch.amp.GradScaler` when `mixed_precision: true`;
CPU disables AMP safely. A finite-loss CUDA FP16 gradient overflow is skipped by
GradScaler, logged as `amp_overflow`, and retried at a lower scale without advancing
the scheduler or global optimizer step. Twenty consecutive overflows abort the run.
Non-finite forward losses remain fatal; FP32 non-finite gradients also remain fatal.
After every successful update, all trainable parameters are checked for finite values.
Stage-1 training may use AMP, but A3 monitor loss runs in FP32 by default
(`monitor_mixed_precision: false`) for numerical robustness.

The real-data CLI refuses to train unless `--mini` or an explicit `--max-steps` is
provided. A Kaggle 50-speaker/50-step smoke command is:

```bash
python -m module_a.scripts.train_model \
  --dataset-root /kaggle/input/datasets/davidthomastran/vietnam-celeb-dataset/full-dataset/data \
  --train-manifest /kaggle/working/module-a-outputs/train_manifest.csv \
  --output-dir /kaggle/working/module-a-a3-mini \
  --device cuda \
  --mini \
  --max-steps 50 \
  --max-train-speakers 50 \
  --max-monitor-speakers 10 \
  --speakers-per-batch 8 \
  --utterances-per-speaker 2 \
  --num-workers 2 \
  --amp
```

Adjust only the manifest path if A1 outputs were written elsewhere. Resume with the
same data/config/limits and add:

```bash
--resume /kaggle/working/module-a-a3-mini/checkpoints/last.pt
```

The run writes `checkpoints/last.pt`, optional step checkpoints, `history.jsonl`,
`run_config.json`, `speaker_to_index.json`, `train_monitor_split.json`, and
`training_summary.json` under the untracked output directory. Resume restores model,
optimizer, scheduler, AMP scaler, epoch/step cursor, and rejects an incompatible
classifier mapping.

Offline A3 verification uses fake WavLM and downloads nothing:

```bash
python -m module_a.scripts.train_model --help
python -m module_a.scripts.sanity_train
```

### Non-finite backward diagnostics

Anomaly tracing is opt-in because it is intentionally slow. `--detect-anomaly` wraps
both the forward and backward pass with `torch.autograd.detect_anomaly(check_nan=True)`
so PyTorch can report the forward operation whose backward first emits NaN/Inf. It
also prints compact first-batch statistics; `--debug-first-step` prints the same
statistics without enabling anomaly tracing. Neither flag changes the default run.

One real non-AMP batch on Kaggle:

```bash
python -m module_a.scripts.train_model \
  --dataset-root /kaggle/input/datasets/davidthomastran/vietnam-celeb-dataset/full-dataset/data \
  --train-manifest /kaggle/working/module-a-outputs/train_manifest.csv \
  --output-dir /kaggle/working/module-a-a3-anomaly \
  --device cuda \
  --mini \
  --max-steps 1 \
  --max-train-speakers 50 \
  --max-monitor-speakers 10 \
  --speakers-per-batch 8 \
  --utterances-per-speaker 2 \
  --num-workers 2 \
  --no-amp \
  --detect-anomaly
```

Use `--amp` instead of `--no-amp` only for the corresponding AMP reproduction.

## A4 speaker-disjoint evaluation

A4 reconstructs the frozen A3 checkpoint and scores only L2-normalized 192-D encoder
embeddings. The AAM classifier is loaded only so the checkpoint can be reconstructed
strictly; it is never used to predict validation or test identities. Embedding
extraction first loads/resamples mono float32 audio to 16 kHz, then evaluates exactly
one configured 3-second segment: long utterances are center-cropped and short
utterances are repeat-padded then truncated to 48,000 samples. Exact-length audio is
unchanged. A4 never uses random crop or zero padding and is not multi-crop or
full-utterance evaluation. Inference uses `model.eval()`, `torch.no_grad()`, and FP32
by default. Per-split `.npz` caches bind the embeddings to the checkpoint,
manifest, evaluation config, dataset root, split, and embedding dimension. An
incompatible cache fails explicitly unless `--recompute-embeddings` is passed.

SV uses bounded same-speaker unordered pairs with no self-pairs. Each speaker gets at
most `max_sv_positive_per_speaker` deterministic positive trials. The protocol then
samples an equal number of unique different-speaker negative pairs using seed 42.
Cosine similarity is the dot product of normalized embeddings. EER is the average of
FAR and FRR at the empirical score threshold minimizing `abs(FAR - FRR)`; it is not
interpolated, and equal objectives prefer the higher threshold. That validation EER
threshold is the frozen deployment SV threshold. Test EER and its threshold are
descriptive only; test FAR/FRR use the persisted validation threshold.

Open-set SID is built independently inside each split. Seeded selection assigns 80%
of speakers (round-half-up, with both sets kept non-empty) to known and the rest to
unknown. Each known speaker contributes at most five deterministic enrollment
utterances and retains at least one disjoint probe; unknown speakers contribute no
enrollment and all their utterances are probes. Enrollment embeddings are averaged
and L2-normalized into prototypes. SID calibration maximizes validation overall
open-set accuracy, where known probes are correct only when accepted with the right
identity and unknown probes are correct only when rejected. Ties prefer the higher,
more conservative threshold.

Validation and test speakers and paths are checked against each other and against the
checkpoint's train-only `speaker_to_index`. Validation is the only phase that writes
`calibration/sv_calibration.json` and `calibration/sid_calibration.json`. Test mode
refuses to start without both persisted validation artifacts, reloads them before
scoring, and never recalibrates or overwrites them. A1 validation/test remain the
primary speaker-disjoint protocol; incomplete official Vietnam-Celeb E/H files are
not used by this A4 implementation.

Validation-only Kaggle command:

```bash
python -m module_a.scripts.evaluate_model \
  --phase validation \
  --dataset-root /kaggle/input/datasets/davidthomastran/vietnam-celeb-dataset/full-dataset/data \
  --val-manifest /kaggle/working/module_a_outputs/val_manifest.csv \
  --checkpoint /kaggle/working/module_a-stage1-full/checkpoints/last.pt \
  --output-dir /kaggle/working/module_a_a4 \
  --device cuda
```

Only after reviewing/fixing the validation protocol and freezing its two calibration
JSON files, run test-only evaluation against the same output directory:

```bash
python -m module_a.scripts.evaluate_model \
  --phase test \
  --dataset-root /kaggle/input/datasets/davidthomastran/vietnam-celeb-dataset/full-dataset/data \
  --val-manifest /kaggle/working/module_a_outputs/val_manifest.csv \
  --test-manifest /kaggle/working/module_a_outputs/test_manifest.csv \
  --checkpoint /kaggle/working/module_a-stage1-full/checkpoints/last.pt \
  --output-dir /kaggle/working/module_a_a4 \
  --device cuda
```

The output tree contains `embeddings/`, `trials/`, `protocols/`, `scores/`,
`calibration/`, `metrics/`, `run_config.json`, and `evaluation_summary.json`. Local
offline verification never downloads WavLM:

```bash
python -m module_a.scripts.evaluate_model --help
python -m module_a.scripts.sanity_evaluation
```

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

### Vietnam-Celeb protocol note

The Kaggle distribution contains the official Vietnam-Celeb `T`, `E`, and `H`
protocol files, but the physically available audio in this Kaggle copy does not fully
cover all referenced paths.

Therefore:

- the primary project protocol uses a reproducible speaker-disjoint 80/10/10 split
  over the audio files physically available in the Kaggle distribution;
- validation is used for checkpoint selection and threshold calibration;
- the test split remains untouched until model and threshold decisions are frozen;
- official `E` and `H` trials may be used later only as supplementary SV evaluation
  after filtering to trials for which both referenced audio files exist;
- any filtered `E`/`H` evaluation must report coverage and excluded-trial counts
  explicitly;
- `vietnam-celeb-t.txt` is not used as the primary training manifest because the
  Kaggle audio copy does not fully cover its referenced utterances.

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
array. A0-A3 does not modify `backend/app/models/speaker_model.py`.

Future exported fields remain `embedding_dimension`, `sv_threshold`, and
`sid_threshold`. Enrollment-quality fields remain `min_duration`, `max_duration`,
`min_speech_ratio`, `min_snr_db`, `max_clipping_ratio`, and `max_content_wer`.

Quality calibration is not part of A1. Later Module A experiments must use the exact
M1 definition: mono 16 kHz PCM16, WebRTC VAD mode 2, complete 30 ms frames, speech
ratio from the same VAD classifications, and VAD-grouped SNR using the same power,
edge-case, and clamping policy. Do not introduce a second SNR estimator.

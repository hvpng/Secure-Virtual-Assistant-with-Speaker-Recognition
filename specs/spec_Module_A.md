# Module A Specification v3 — ECAPA-TDNN on VoxVietnam

## 1. Goal

Implement one representative speaker-recognition model for the project rubric:

- Model: **ECAPA-TDNN**
- Dataset: **VoxVietnam**
- Training data: **VoxVietnam-T**
- Validation: speaker-disjoint split carved from VoxVietnam-T
- Final test: **VoxVietnam-O**
- Tasks supported by the same speaker embedding model:
  - Speaker Verification (SV)
  - Speaker Identification (SID)
- Final artifact must be exportable and usable by the existing web application.

The implementation should be simple enough that the user can:

1. push the code,
2. open one Kaggle notebook,
3. run all cells,
4. obtain the trained model, evaluation results, and export artifact.

Do not implement multiple competing model families.

---

## 2. Scope

### Primary model

Use **ECAPA-TDNN** as a dedicated speaker embedding model.

Expected inference pipeline:

```text
16 kHz waveform
→ 80-bin log Mel filterbank / fbank
→ ECAPA-TDNN
→ 192-dimensional speaker embedding
→ L2 normalization
```

During training only:

```text
embedding
→ AAM-Softmax classification head
→ train-speaker classification loss
```

The AAM-Softmax classification head is not required for deployment.

### Model output contract

The final embedding must be:

- 1-D
- shape `(192,)`
- `numpy.float32`
- finite
- L2-normalized

---

## 3. Dataset

Use the official **VoxVietnam** dataset distribution.

Preferred source of truth:

```text
hustep-lab/VoxVietnam-Dataset
```

The implementation must support either:

1. Hugging Face access with a user-provided token, or
2. an already-mounted local/Kaggle dataset directory.

Do not hardcode a single Kaggle input path.

### Dataset roles

```text
VoxVietnam-T
├── train split
└── validation split

VoxVietnam-O
└── final test only
```

### Train/validation split

Create a deterministic **speaker-disjoint 90/10 split** from VoxVietnam-T.

Requirements:

- seed = 42
- no speaker overlap between train and validation
- train speakers are used by the AAM-Softmax classifier
- validation speakers are unseen during classification training

Do not use VoxVietnam-O for:

- hyperparameter tuning
- checkpoint selection
- threshold calibration
- architecture decisions

VoxVietnam-O is final evaluation only.

---

## 4. Audio and feature processing

All audio must be converted/normalized to:

- mono
- 16 kHz
- floating-point waveform

Training segment:

- random 3.0-second crop
- short utterances may be repeated and then truncated to 3.0 seconds

Validation/final evaluation:

- deterministic segment preparation
- no random crop

Feature extraction:

- 80-bin log Mel filterbank / fbank
- per-utterance mean normalization

The same feature implementation must be used for train, validation, test, and exported inference.

---

## 5. Phase 1 — Train and validate

Implement one training script:

```bash
python -m module_a.scripts.train_ecapa
```

It must:

1. load/index VoxVietnam-T,
2. create or reuse deterministic train/validation manifests,
3. initialize ECAPA-TDNN,
4. train using AAM-Softmax,
5. periodically evaluate speaker embeddings on validation speakers,
6. select the best checkpoint by **validation SV EER**,
7. calibrate validation thresholds after training,
8. save a compact training summary.

### Training defaults

Use practical Kaggle-T4 defaults, configurable from one YAML file.

Recommended initial defaults:

```yaml
sample_rate: 16000
segment_seconds: 3.0
fbank_dim: 80
embedding_dim: 192

loss:
  name: aam_softmax
  margin: 0.2
  scale: 30.0

optimizer:
  name: adamw
  lr: 0.001
  weight_decay: 0.0001

training:
  epochs: 20
  batch_size: 32
  gradient_accumulation: 2
  amp: true
  num_workers: 2

augmentation:
  speed_perturb: true
  additive_noise: true
  reverb: true
```

If noise/reverb resources are unavailable, the pipeline must degrade gracefully to available augmentation and record what was actually enabled in the summary.

### Sampling

Training must use speaker-aware or speaker-balanced sampling so speakers with many utterances do not dominate training.

### Checkpointing

Save at least:

```text
last.pt
best.pt
```

Checkpoint metadata must include:

- model config
- training config
- epoch
- global step
- speaker-to-index mapping
- optimizer state for resumable training
- validation metric used for best-checkpoint selection

---

## 6. Validation evaluation

Validation speakers are unseen speakers.

### Speaker Verification

Generate deterministic validation trials:

- positive: same-speaker pairs
- negative: different-speaker pairs
- cosine similarity over L2-normalized embeddings

Report:

- EER
- AUC
- FAR
- FRR
- threshold at target FAR
- minDCF if implemented reliably

Default deployment target:

```text
SV target FAR = 0.05
```

Calibrate the SV threshold on validation only.

### Speaker Identification

Use the same ECAPA embeddings.

Construct a deterministic open-set SID validation protocol:

- split validation speakers into known and unknown groups
- known speakers:
  - enrollment utterances
  - probe utterances
- unknown speakers:
  - probe utterances only
- enrollment prototype:
  - mean enrollment embeddings
  - L2-normalize the mean

Report:

- known-speaker top-1 identity accuracy
- known accepted-correct rate
- known rejection rate
- known wrong-accept rate
- unknown false-accept rate
- unknown rejection rate

Calibrate the SID threshold on validation only.

Default:

```text
SID target unknown FAR = 0.05
```

---

## 7. Phase 2 — Final evaluation and export

Implement one script:

```bash
python -m module_a.scripts.evaluate_export
```

It must:

1. load `best.pt`,
2. load frozen validation calibration,
3. evaluate the model on VoxVietnam-O,
4. never recalibrate thresholds from VoxVietnam-O,
5. save final metrics,
6. export the deployment artifact.

### Final SV evaluation

If an official VoxVietnam-O verification trial list is available, use it.

Do not silently replace an official protocol with generated trials.

If the dataset source does not provide the required official trial list, fail with a clear message or explicitly run a documented custom protocol. Never label a custom protocol as the official benchmark.

Report at least:

- EER
- AUC
- FAR at frozen validation threshold
- FRR at frozen validation threshold
- TPR/TAR at frozen validation threshold
- minDCF if supported

### Final SID evaluation

Use a clearly documented custom SID protocol on VoxVietnam-O.

Do not claim custom SID metrics are official VoxVietnam benchmark results.

---

## 8. Export artifact

Output:

```text
module_a_export/
├── model.pt
├── config.json
├── thresholds.json
└── metadata.json
```

`thresholds.json` must contain thresholds calibrated on validation, not final test.

The runtime ABI must be:

```python
load_model(model_dir, device="auto")
extract_embedding(model, audio_path)
```

`extract_embedding(...)` must return:

- `np.ndarray`
- dtype `np.float32`
- shape `(192,)`
- all finite
- L2 norm approximately 1.0

No training dependencies should be required by the web application beyond what is needed for inference.

---

## 9. Required project outputs

At the end of a successful Kaggle Run All, produce:

```text
outputs/
├── manifests/
│   ├── train.csv
│   └── validation.csv
├── checkpoints/
│   ├── last.pt
│   └── best.pt
├── calibration/
│   ├── sv_calibration.json
│   └── sid_calibration.json
├── metrics/
│   ├── validation_metrics.json
│   ├── sv_test_metrics.json
│   └── sid_test_metrics.json
├── training_summary.json
├── evaluation_summary.json
└── module_a_export/
```

---

## 10. Kaggle workflow

Provide one notebook or copy-paste-ready sequence with approximately five cells:

### Cell 1
Clone/pull repository and install dependencies.

### Cell 2
Set:

- project directory
- Hugging Face token if needed
- dataset source/path
- output directory

### Cell 3
Run Phase 1 training.

### Cell 4
Run Phase 2 final evaluation and export.

### Cell 5
Print compact summaries and list exported files.

The user should not need to manually edit Python source code inside Kaggle.

---

## 11. Code organization

Rebuild `module_a/` around the new primary implementation.

Suggested minimal structure:

```text
module_a/
├── SPEC.md
├── README.md
├── configs/
│   └── ecapa_voxvietnam.yaml
├── scripts/
│   ├── train_ecapa.py
│   └── evaluate_export.py
├── src/
│   ├── data.py
│   ├── features.py
│   ├── ecapa.py
│   ├── training.py
│   ├── evaluation.py
│   └── runtime.py
└── tests/
```

Avoid unnecessary abstraction.

---

## 12. Legacy Module A policy

The previous WavLM + CAM++ implementation is no longer the primary Module A.

Delete the old `module_a/` implementation from the current working tree and rebuild it cleanly according to this specification.

Do not preserve old WavLM/CAM++ code under `module_a/legacy/`.

Git history already preserves the previous implementation.

Before deletion, ensure no Track B code imports old Module A internals. The web application must integrate only through the final exported runtime ABI.

---

## 13. Acceptance criteria

The rewrite is accepted only when all are true:

1. `module_a/` contains only the ECAPA + VoxVietnam primary pipeline.
2. train/validation split is deterministic and speaker-disjoint.
3. VoxVietnam-O is not touched during Phase 1.
4. training runs on CUDA with AMP when available.
5. best checkpoint is selected by validation SV EER.
6. validation thresholds are persisted.
7. Phase 2 consumes frozen validation thresholds.
8. final test never recalibrates thresholds.
9. SV and SID both use the same ECAPA embedding model.
10. exported embeddings are finite, float32, 192-D, and L2-normalized.
11. unit tests and smoke tests pass.
12. Kaggle instructions are copy-paste-ready and support Run All.
13. README/report notes clearly distinguish:
    - validation results,
    - final VoxVietnam-O SV results,
    - custom SID protocol results,
    - published reference results from the VoxVietnam paper.

---

## 14. Report framing

The final report only needs to clearly cover the rubric:

- Dataset: VoxVietnam
- Split: VoxVietnam-T train/validation + VoxVietnam-O final test
- Model: ECAPA-TDNN
- Training procedure: 80-d fbank, 3 s segments, augmentation, AAM-Softmax, optimizer/settings
- SV metrics: EER, AUC, FAR/FRR, optionally minDCF
- SID metrics: top-1 accuracy and open-set acceptance/rejection metrics
- Experimental results
- Comparison with published VoxVietnam ECAPA reference only when the evaluation protocol is equivalent

Do not claim SOTA unless the experiment genuinely establishes it.

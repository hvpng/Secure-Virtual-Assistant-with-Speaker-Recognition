# MODULE A — Speaker Model Training, Evaluation & Export

> Chạy trên Kaggle.  
> Module A là training/evaluation workspace riêng, không thuộc runtime application của Track B.  
> Track B chỉ nhận artifact inference cuối cùng theo stable ABI.

---

## 1. Mục tiêu

Train/evaluate một speaker recognition model phục vụ cả:

- Speaker Verification (SV)
- Speaker Identification (SID)

Sau đó export artifact inference ổn định để Module B có thể sử dụng mà không phụ thuộc trực tiếp vào training notebook hay architecture nội bộ.

Primary architecture:

```text
16 kHz mono audio
 -> WavLM-Base+
 -> LayerNorm
 -> trainable Linear projection 768 -> 80
 -> CAM++
 -> speaker embedding
 -> cosine similarity
```

Projection `768 -> 80` là adapter engineering của nhóm để đưa WavLM hidden representation vào CAM++ interface 80-dimensional gần recipe Fbank.

Phải document rõ:
- đây là thiết kế của nhóm;
- không claim đây là recipe nguyên bản từ paper/reference nếu không có bằng chứng trực tiếp.

---

## 2. Module boundaries

Module A nằm riêng trong:

```text
module_a/
```

Không đặt training loop, augmentation, dataset loader hoặc experiment code trong:

```text
backend/app/models/
```

Backend chỉ nhận artifact cuối:

```text
backend/app/models/
├── speaker_model.py
├── speaker_config.json
├── enrollment_config.json
└── weights/
```

Module A không được sửa auth logic của M1–M6.

---

## 3. Milestones bắt buộc

Module A được chia thành:

```text
A0 — structure + config + stable contracts
A1 — dataset inspection + manifest + split
A2 — model forward/backward sanity
A3 — training + checkpointing
A4 — SV/SID evaluation + calibration
A5 — export + local ABI smoke
```

Rule:

```text
KHÔNG full-train trên Kaggle trước khi A0, A1, A2 PASS.
```

Mỗi milestone phải có sanity checks riêng.

---

## 4. Dataset

Primary dataset:

```text
Vietnam-Celeb
```

Nếu có thể truy cập hợp lệ:

```text
VLSP2021-SV
```

có thể dùng như cross-dataset test bổ sung.

Không phụ thuộc VLSP2021-SV để hoàn thành project.

---

## 5. Dataset inspection bắt buộc

Trước khi split/train, phải ghi nhận ít nhất:

- dataset root/path
- số speaker
- số utterance
- số utterance mỗi speaker
- sample rate distribution
- duration statistics
- file format distribution
- corrupted/unreadable files
- duplicate paths nếu có

Phải có dataset inspection output reproducible.

Không giả định layout trước khi inspect dataset thật trên Kaggle.

---

## 6. Manifest

Tạo manifest canonical, ví dụ:

```csv
path,speaker_id,split
/path/a.wav,spk001,train
/path/b.wav,spk001,train
/path/c.wav,spk051,val
/path/d.wav,spk080,test
```

Manifest phải:
- lưu chính xác;
- deterministic;
- dùng fixed seed;
- có thể dùng lại cho training/evaluation/report.

Không tạo split ngẫu nhiên lại mỗi lần notebook chạy.

---

## 7. Split policy

Primary split:

```text
80% speakers train
10% speakers validation
10% speakers test
seed = 42
```

Speaker-disjoint:

```text
train speakers ∩ val speakers = ∅
train speakers ∩ test speakers = ∅
val speakers ∩ test speakers = ∅
```

Mục tiêu:

- train speaker-discriminative embedding trên train speakers;
- đánh giá generalization trên unseen validation/test speakers.

SID trong Module A là:

```text
open-set speaker identification / recognition trên unseen speakers
```

không phải classifier inference bằng training class ID.

Classifier head nếu có chỉ dùng trong training objective.

---

## 8. Audio preprocessing

Canonical input:

```text
mono
16 kHz
PCM-compatible waveform
```

Training/evaluation phải xử lý:

- stereo -> mono
- non-16k -> resample 16k
- invalid/corrupt audio -> controlled skip/error
- duration cropping/padding theo config

Không normalize theo cách làm thay đổi speaker characteristics quá mạnh.

Production `extract_embedding()` phải nhận 16 kHz mono WAV như Track B contract.

---

## 9. Training tiers

### Tier 1 — Primary

```text
WavLM-Base+
 -> adapter/projection
 -> CAM++
```

### Stage 1

- freeze toàn bộ WavLM;
- train adapter + CAM++;
- AAM-Softmax hoặc objective speaker classification đã chốt;
- augmentation noise/reverb nếu pipeline đã ổn;
- mixed precision nếu GPU hỗ trợ.

Đây là baseline bắt buộc phải thử trước.

### Stage 2 — Optional

Chỉ khi Stage 1 chạy ổn và còn thời gian:

- unfreeze một số layer cuối WavLM;
- LR WavLM nhỏ hơn backend;
- log rõ layer nào unfreeze;
- so sánh validation metric với Stage 1.

Không unfreeze toàn bộ WavLM ngay từ đầu.

---

## 10. Fallback tiers

### Tier 2

```text
80-dim Fbank
 -> CAM++
```

### Tier 3

```text
pretrained ECAPA-TDNN
 -> fine-tune tiếng Việt
```

Fallback chỉ dùng nếu Tier 1 gặp blocker thực tế như:

- OOM liên tục;
- dependency/runtime blocker;
- training instability không thể khắc phục trong deadline.

Dù dùng tier nào, artifact cuối vẫn phải export cùng ABI.

---

## 11. Model output

Model phải có speaker embedding trước classification head.

Embedding production phải:

- `np.ndarray`
- 1-D
- `float32`
- finite
- fixed dimension
- L2-normalized
- deterministic trong eval mode

Cosine similarity là scoring function canonical cho Module B.

---

## 12. Training objective

Primary objective:

```text
AAM-Softmax
```

Nếu implementation thực tế cần classification head:

```text
embedding
 -> classifier / AAM head
 -> training logits
```

Classifier head chỉ phục vụ training.

Production inference không dùng class logits để xác định employee.

---

## 13. Checkpoint policy

Lưu tối thiểu:

```text
best.pt
last.pt
history.csv
```

Best checkpoint phải chọn bằng validation metric.

Không chọn best checkpoint theo test result.

Checkpoint phải đủ để:
- resume training;
- evaluate độc lập;
- export artifact.

---

## 14. SV evaluation protocol

Tạo:

```text
verification_trials.csv
```

reproducible.

Positive pair:
- hai utterance khác nhau;
- cùng speaker.

Negative pair:
- hai utterance;
- khác speaker.

Không dùng cùng file cho positive pair.

Nếu số pair nhỏ:
- dùng toàn bộ positive pairs.

Nếu quá lớn:
- deterministic cap;
- seed = 42;
- ghi cap trong report.

Negative pairs:
- sample deterministic;
- ưu tiên số lượng bằng positive pairs nếu quy mô hợp lý.

---

## 15. SV scoring

Mỗi utterance:

```text
audio
 -> embedding
```

Pair score:

```text
cosine_similarity(embedding_a, embedding_b)
```

Không dùng classifier probability làm SV score.

---

## 16. SV metrics

Bắt buộc:

```text
EER
```

Khuyến khích:

```text
minDCF
FAR
FRR
ROC/DET
```

Validation:
- chọn checkpoint;
- calibrate `sv_threshold`.

Test:
- chỉ chạy sau khi threshold freeze;
- không tune lại threshold.

---

## 17. SV threshold calibration

Threshold phải lấy từ validation scores.

Canonical policy có thể là:

```text
threshold gần EER operating point
```

hoặc một operating point khác nếu report giải thích rõ.

Phải lưu:

```text
validation threshold
validation FAR
validation FRR
validation EER
```

Sau khi freeze:

```text
SV_THRESHOLD = fixed value
```

Không thay đổi bằng test results.

---

## 18. SID evaluation protocol

Validation/test speaker split tiếp tục được dùng theo open-set protocol.

Với mỗi validation hoặc test split:

1. fixed seed chia speakers thành:
   - known/gallery: 80%
   - unknown: 20%.

2. Với mỗi known speaker:
   - chọn tối đa 5 utterances làm enrollment;
   - enrollment utterances phải khác probe;
   - mean embedding -> speaker prototype.

3. Remaining known utterances:
   - known probes.

4. Unknown speakers:
   - chỉ làm unknown probes.

Seed:

```text
42
```

Phải reproducible.

---

## 19. SID scoring

Known prototypes:

```text
prototype_speaker = mean(enrollment_embeddings)
prototype_speaker = L2 normalize
```

Probe:

```text
probe_embedding
 -> cosine với tất cả prototypes
 -> max score
 -> predicted speaker
```

Decision:

```text
if max_score >= sid_threshold:
    accept predicted speaker
else:
    unknown
```

Không dùng classifier label từ train speakers.

---

## 20. SID metrics

Bắt buộc:

- top-1 accuracy trên known probes;
- unknown rejection performance.

Khuyến khích:

- FAR;
- FRR;
- EER known-vs-unknown;
- confusion summary nếu hợp lý.

`sid_threshold` chỉ calibrate trên validation.

Test chỉ dùng threshold đã freeze.

---

## 21. Enrollment quality experiments

Quality threshold calibration dùng:

```text
validation only
```

Không dùng test.

### Duration

Evaluate degraded/enrollment duration:

```text
8.0 s
5.0 s
3.0 s
1.5 s
```

Có thể bổ sung point khác nếu cần.

Dùng degradation để quyết định:

```text
min_duration
max_duration
```

### SNR

Evaluate:

```text
20 dB
15 dB
10 dB
5 dB
0 dB
```

Dùng degradation để chọn:

```text
min_snr_db
```

---

## 22. Quality metric definitions — PHẢI KHỚP M1

Module A phải dùng cùng definition với M1.

### Canonical VAD

```text
WebRTC VAD
mode = 2
frame = 30 ms
sample rate = 16 kHz
mono PCM16
```

Trailing incomplete frame:

```text
ignore
```

### Speech ratio

```text
speech_frames / valid_frames
```

theo cùng `_classify_vad_frames()` logic của M1.

### SNR

Speech power:

```text
mean squared PCM16 sample values
trên speech frames
```

Noise power:

```text
mean squared PCM16 sample values
trên non-speech frames
```

Canonical:

```text
SNR = 10 * log10(
    speech_power / max(noise_power, 1 PCM^2)
)
```

Edge cases:
- missing speech group -> deterministic low/fail value;
- missing noise group -> deterministic controlled value;
- zero speech power -> 0 dB policy như M1;
- no NaN/inf;
- clamp range giống M1.

Module A không được dùng một SNR estimator khác để calibrate threshold rồi áp sang backend M1.

---

## 23. Other enrollment quality thresholds

Config cuối phải có:

```text
min_duration
max_duration
min_speech_ratio
min_snr_db
max_clipping_ratio
max_content_wer
```

Nếu threshold:
- derived từ experiment -> source `"validation_experiment"`
- heuristic -> source `"engineering_default"`

Engineering default phải được M7 validate bằng real enrollment.

---

## 24. Content matching

`max_content_wer` liên quan đến PhoWhisper/M2 content checking.

Module A không cần retrain ASR.

Nếu không chạy được experiment WER đầy đủ trong Module A:
- giữ engineering default;
- ghi rõ source;
- validate lại trong M7.

Không claim threshold là experiment-derived nếu không thực sự đo.

---

## 25. Enrollment scripts

Canonical 7 scripts:

1. Hôm nay tôi muốn kiểm tra lịch họp của mình.
2. Xin chào, tôi là nhân viên mới của phòng kỹ thuật.
3. Giá vé máy bay tháng này đã tăng lên rất nhiều.
4. Chúng ta nên nghỉ ngơi một chút trước khi tiếp tục công việc.
5. Ông bà tôi sống ở một ngôi làng nhỏ gần biển.
6. Cô ấy vừa mua một chiếc xe đạp màu đỏ rất đẹp.
7. Trời hôm nay se lạnh, gió thổi nhẹ qua khung cửa sổ.

Backend M4/M5 phải dùng cùng source of truth khi final integration.

---

## 26. Enrollment-script research reference

Paper chính:

```text
Phoneme-Based Optimization of Enrollment Selection for Speaker Identification
DOI 10.1007/978-981-95-7075-1_34
```

Correction:

```text
DOI 10.1007/978-981-95-7075-1_51
```

Nếu nhóm không reproduce đầy đủ thuật toán:
- report phải nói rõ 7 câu là thiết kế của nhóm;
- paper chỉ dùng làm motivation cho phonetic coverage.

Không claim exact reproduction.

---

## 27. Stable export directory

Module A export:

```text
export/
├── speaker_model.py
├── weights/
│   └── speaker_model.pt
├── speaker_config.json
├── enrollment_config.json
├── enrollment_config.md
├── enrollment_scripts.md
├── verification_trials.csv
├── sid_protocol.csv
├── eval_results.csv
├── calibration_results.csv
└── model_selection.md
```

Tên weight file có thể khác nếu implementation cần, nhưng phải document rõ.

---

## 28. Stable application ABI

`speaker_model.py` phải expose:

```python
load_model(
    model_dir: str,
    device: str = "auto",
)

extract_embedding(
    model,
    audio_path: str,
) -> np.ndarray
```

`load_model()`:
- không phụ thuộc notebook globals;
- không cần training dataset;
- không tự train;
- không silent fallback;
- load artifact deterministic.

`extract_embedding()`:
- nhận 16 kHz mono WAV;
- return 1-D `np.float32`;
- finite;
- fixed dimension;
- L2-normalized.

---

## 29. Runtime dependency policy

Final artifact không được phụ thuộc vào:

- Kaggle-only filesystem;
- notebook variable;
- notebook mounted dataset;
- training optimizer;
- training dataloader.

Nếu WavLM pretrained weights cần runtime:
- ưu tiên package/export strategy rõ ràng;
- không để final demo bất ngờ download model lớn nếu có thể tránh.

Nếu vẫn cần Hugging Face cache/model ID:
- document dependency chính xác trong export README/model_selection;
- M7 phải smoke test trên Windows trước demo.

---

## 30. `speaker_config.json`

Canonical schema:

```json
{
  "architecture": "wavlm_base_plus_campp",
  "tier": 1,
  "sample_rate": 16000,
  "embedding_dimension": 192,
  "sv_threshold": 0.0,
  "sid_threshold": 0.0,
  "threshold_source": "validation",
  "seed": 42
}
```

`embedding_dimension`, `sv_threshold`, `sid_threshold` ở trên chỉ là ví dụ schema.

Trước M7:
- tất cả phải là giá trị thật;
- không để placeholder 0.0;
- dimension phải khớp model thật.

Field name canonical cho Track B:

```text
embedding_dimension
```

Không dùng lẫn `embedding_dim` và `embedding_dimension`.

---

## 31. `enrollment_config.json`

Canonical schema:

```json
{
  "min_duration": 1.5,
  "max_duration": 8.0,
  "min_speech_ratio": 0.0,
  "min_snr_db": 0.0,
  "max_clipping_ratio": 0.001,
  "max_content_wer": 0.30,
  "sources": {
    "min_duration": "validation_experiment",
    "max_duration": "engineering_default",
    "min_speech_ratio": "engineering_default",
    "min_snr_db": "validation_experiment",
    "max_clipping_ratio": "engineering_default",
    "max_content_wer": "engineering_default"
  }
}
```

Các số trên chỉ minh họa schema.

Trước M7:
- thay bằng decision thật;
- không copy placeholder một cách mù quáng.

Field names phải khớp M1 exactly.

---

## 32. Training outputs bắt buộc

Lưu đủ để viết report:

```text
dataset_summary.json
train_manifest.csv
val_manifest.csv
test_manifest.csv
training_history.csv
best checkpoint
last checkpoint
verification_trials.csv
sid_protocol.csv
validation metrics
test metrics
calibration results
```

Report phải có thể reconstruct:
- dataset split;
- model;
- training configuration;
- checkpoint selection;
- threshold calibration;
- final test metrics.

---

## 33. Reproducibility

Fixed seed mặc định:

```text
42
```

Seed tối thiểu cho:
- Python random;
- NumPy;
- PyTorch;
- split;
- pair sampling;
- SID known/unknown split.

Nếu CUDA deterministic mode làm training quá chậm:
- document exact compromise;
- vẫn giữ deterministic data split/evaluation protocols.

---

## 34. Kaggle workflow

Notebook execution order:

```text
1. environment + paths
2. config
3. dataset inspection
4. manifest creation/load
5. data loader sanity
6. model construction
7. forward sanity
8. backward sanity
9. mini training
10. full training
11. validation
12. SV calibration
13. SID calibration
14. frozen test evaluation
15. export
```

Không chạy full training ngay cell đầu.

---

## 35. A2 sanity gate

Trước A3 phải chứng minh:

- 1 audio load được;
- 1 batch load được;
- WavLM output shape đúng;
- adapter output shape đúng;
- CAM++ forward đúng;
- embedding fixed dimension;
- loss finite;
- backward chạy;
- optimizer step chạy;
- checkpoint save/load chạy;
- không OOM trên mini batch.

Nếu fail:
- sửa trước khi full train.

---

## 36. Mini-training gate

Trước full training:

Chạy nhỏ, ví dụ:
- subset speakers;
- subset utterances;
- 1 epoch hoặc vài trăm steps.

Phải thấy:
- loss không NaN;
- loss có xu hướng hợp lý;
- checkpoint tạo được;
- validation pipeline chạy được.

Không yêu cầu metric đẹp ở mini run.

---

## 37. Model selection

`model_selection.md` phải ghi:

- Tier nào thực tế dùng;
- Stage nào dùng;
- WavLM frozen/unfrozen ra sao;
- best checkpoint criterion;
- validation metric;
- lý do chọn checkpoint;
- nếu fallback tier, lý do fallback.

Không chọn architecture bằng test set.

---

## 38. Final test rule

Test set chỉ chạy sau khi:

```text
best checkpoint frozen
SV threshold frozen
SID threshold frozen
quality config frozen
```

Không:
- retune architecture;
- retune threshold;
- đổi split;
- đổi quality threshold
sau khi xem test result.

Nếu phát hiện bug implementation:
- fix;
- invalidate affected test result;
- rerun theo protocol;
- document.

---

## 39. Local export smoke

Trước M7, artifact phải được copy/download về local Windows.

Smoke test:

```python
model = load_model(model_dir, device="auto")
embedding = extract_embedding(model, wav_path)
```

Assert:

```text
embedding.ndim == 1
embedding.dtype == float32
all finite
fixed length
L2 norm approximately 1
```

Chạy ít nhất 2 lần cùng file:

```text
cosine(embedding_run1, embedding_run2) ≈ 1
```

Eval mode phải deterministic.

---

## 40. Track B handoff

Khi Module A PASS:

Copy canonical artifact vào:

```text
backend/app/models/
```

Sau đó M7:

```text
SPEAKER_BACKEND=real
```

Không sửa M1 ABI trừ blocker thật.

Không thêm silent fake fallback.

---

## 41. Gate bàn giao Module A

Module A chỉ PASS khi:

- [ ] exact train/val/test manifests đã lưu
- [ ] dataset statistics đã lưu
- [ ] Tier thực tế được document
- [ ] checkpoint train thật tồn tại
- [ ] SV evaluation hoàn tất
- [ ] SID evaluation hoàn tất
- [ ] `sv_threshold` calibrate trên validation
- [ ] `sid_threshold` calibrate trên validation
- [ ] quality config có source rõ
- [ ] final test chạy sau calibration freeze
- [ ] `speaker_model.py` import được trên Windows
- [ ] `load_model()` không cần notebook state
- [ ] `extract_embedding()` đúng stable ABI
- [ ] embedding 1-D float32 finite fixed-dim L2-normalized
- [ ] eval deterministic
- [ ] config không còn placeholder
- [ ] artifact load được bằng Track B
- [ ] không silent fallback sang fake backend

---

## 42. Definition of DONE

```text
Vietnam-Celeb
 -> reproducible split
 -> trained speaker embedding model
 -> SV metrics
 -> SID metrics
 -> validation calibration
 -> frozen final test
 -> stable exported artifact
 -> local Track B ABI smoke PASS
```

Module A chưa DONE nếu chỉ:

```text
"training notebook chạy"
```

hoặc:

```text
"checkpoint được save"
```

Mục tiêu cuối là artifact thật có thể cắm vào application.

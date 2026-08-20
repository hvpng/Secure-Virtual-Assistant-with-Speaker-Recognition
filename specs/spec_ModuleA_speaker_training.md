# MODULE A — Speaker Model Training & Evaluation

> Chạy trên Kaggle. Không thuộc phạm vi Codex local.

## 1. Mục tiêu

Train/evaluate model speaker recognition phục vụ cả SV và SID, sau đó export application ABI ổn định để Module B không phụ thuộc architecture cụ thể.

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

Projection 768 -> 80 là adapter engineering để đưa WavLM hidden representation vào CAM++ interface 80-dimensional gần recipe Fbank. Phải document đây là thiết kế của nhóm, không claim là recipe công bố sẵn.

## 2. Dataset/split

Primary: Vietnam-Celeb.

Speaker-disjoint:
- 80% speakers train
- 10% speakers validation
- 10% speakers test
- fixed seed = 42
- lưu manifest chính xác.

Cross-dataset:
- VLSP2021-SV test set nếu truy cập được hợp lệ.

Không dùng test set để chọn checkpoint, threshold, architecture hay quality config.

## 3. Training tiers

### Tier 1 — mục tiêu
WavLM-Base+ -> adapter -> CAM++.

Stage 1:
- freeze toàn bộ WavLM;
- train adapter + CAM++;
- AAM-Softmax;
- augmentation noise/reverb nếu pipeline ổn.

Stage 2 — optional
Chỉ khi Stage 1 ổn và còn thời gian:
- unfreeze một số layer cuối WavLM;
- learning rate WavLM nhỏ hơn backend;
- log rõ layer nào được unfreeze.

### Tier 2
80-dim Fbank -> CAM++.

### Tier 3
ECAPA-TDNN pretrained -> fine-tune tiếng Việt.

**Tier nào thắng cũng phải export cùng `speaker_model.py` interface.**

## 4. SV evaluation protocol

Tạo `verification_trials.csv` reproducible:
- positive pair: hai utterance cùng speaker;
- negative pair: hai utterance khác speaker;
- dùng tất cả positive pairs nếu quy mô hợp lý;
- sample số negative pairs bằng số positive pairs với seed 42;
- nếu quá lớn, cap deterministic và ghi cap trong report.

Metrics:
- EER bắt buộc;
- minDCF khuyến khích;
- DET curve nếu kịp.

Validation:
- chọn best checkpoint;
- calibration `sv_threshold`.

Test:
- chỉ chạy sau khi threshold freeze.

## 5. SID evaluation protocol

Trên validation/test:
1. fixed seed chia speakers trong split thành:
   - known/gallery 80%;
   - unknown 20%.
2. Với mỗi known speaker:
   - tối đa 5 utterance làm enrollment;
   - mean embedding;
   - utterance còn lại làm probe.
3. Unknown speakers chỉ làm probe.

Metrics:
- top-1 accuracy trên known probes;
- unknown rejection;
- FAR/FRR hoặc EER cho known-vs-unknown.

`sid_threshold` calibration chỉ trên validation rồi freeze.

## 6. Enrollment quality experiments

Dùng **validation**, không dùng test.

### SNR
20, 15, 10, 5, 0 dB -> đo degradation -> chọn `min_snr_db`.

### Duration
8, 5, 3, 1.5 s -> đo degradation -> chọn `min_duration_sec`.

Mọi quality threshold vẫn phải nằm trong Module A config:
- `max_duration_sec`
- `min_speech_ratio`
- `max_clipping_ratio`
- `max_content_wer`

Nếu là engineering heuristic thay vì experiment-derived, ghi `"source": "engineering_default"` và validate ở M7.

## 7. Enrollment scripts

1. Hôm nay tôi muốn kiểm tra lịch họp của mình.
2. Xin chào, tôi là nhân viên mới của phòng kỹ thuật.
3. Giá vé máy bay tháng này đã tăng lên rất nhiều.
4. Chúng ta nên nghỉ ngơi một chút trước khi tiếp tục công việc.
5. Ông bà tôi sống ở một ngôi làng nhỏ gần biển.
6. Cô ấy vừa mua một chiếc xe đạp màu đỏ rất đẹp.
7. Trời hôm nay se lạnh, gió thổi nhẹ qua khung cửa sổ.

Paper chính:
- *Phoneme-Based Optimization of Enrollment Selection for Speaker Identification*
- DOI `10.1007/978-981-95-7075-1_34`

Correction:
- DOI `10.1007/978-981-95-7075-1_51`

Nếu không reproduce đầy đủ thuật toán, report phải nói rõ 7 câu là thiết kế của nhóm.

## 8. Stable export contract

```text
export/
├── speaker_model.py
├── weights/
├── speaker_config.json
├── enrollment_config.json
├── enrollment_config.md
├── enrollment_scripts.md
├── verification_trials.csv
├── eval_results.csv
└── model_selection.md
```

### `speaker_model.py`

```python
load_model(model_dir: str, device: str = "auto")
extract_embedding(model, audio_path: str) -> np.ndarray
```

### `speaker_config.json`

Tối thiểu:

```json
{
  "architecture": "wavlm_base_plus_campp",
  "sample_rate": 16000,
  "embedding_dim": 0,
  "sv_threshold": 0.0,
  "sid_threshold": 0.0,
  "threshold_source": "validation"
}
```

`embedding_dim=0` và threshold `0.0` là placeholder; phải thay số thật trước M7.

### `enrollment_config.json`

```json
{
  "min_duration_sec": 0.0,
  "max_duration_sec": 8.0,
  "min_speech_ratio": 0.0,
  "min_snr_db": 0.0,
  "max_clipping_ratio": 0.001,
  "max_content_wer": 0.30
}
```

Các placeholder phải thay bằng decision thật trước M7.

## 9. Gate bàn giao

- [ ] `speaker_model.py` import được trên Windows.
- [ ] `load_model()` không cần notebook state.
- [ ] `extract_embedding()` nhận 16 kHz mono WAV.
- [ ] threshold không placeholder.
- [ ] embedding finite, fixed dimension.
- [ ] eval mode deterministic.
- [ ] ghi rõ Tier thực tế.
- [ ] final test chạy sau calibration freeze.

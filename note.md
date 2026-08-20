# Phân tích chi tiết 3 đề Đồ án cuối kỳ

> Kết hợp: (1) nội dung chính thức từ 3 file đề bài PDF, (2) ghi chú buổi họp đã tổng hợp trước đó, (3) đánh giá/phân tích thêm về implementation, thời gian, rủi ro.

---

## Quy định chung cho cả 3 đề

- **Nộp bài:** đóng gói toàn bộ source code, model weights đã train, dataset dùng để train, và báo cáo vào **một file ZIP**.
- **Tên file:** `StudentID1_StudentID2_..._StudentIDN.zip` (mã số sinh viên các thành viên, cách nhau bởi dấu gạch dưới).
- **Nếu file quá lớn:** upload lên Google Drive, nộp kèm file `.txt` chứa link, đặt tên theo cùng quy tắc (`StudentID1_..._StudentIDN.txt`).
- **Phân bổ điểm khác nhau giữa 3 đề** — đây là điểm quan trọng cần lưu ý khi phân bổ effort:

| Đề | Phần model/thực nghiệm | Phần ứng dụng/web |
|---|---|---|
| 1 — Object Detection | 8 điểm | 2 điểm |
| 2 — NLP Transformer | 8 điểm | 2 điểm |
| 3 — Voice Assistant | 5 điểm | 5 điểm |

→ Đề 1 và 2: **trọng tâm dồn gần như tuyệt đối vào chất lượng model/thực nghiệm**, phần web chỉ là demo cho có. Đề 3: **cân bằng**, phần tích hợp ứng dụng quan trọng ngang phần model.

---

## ĐỀ 1 — Object Detection Application

### 1. Yêu cầu chính thức từ đề bài

**Bối cảnh:** Object detection có nhiều ứng dụng thực tế (giám sát an ninh, phát hiện phương tiện, kiểm tra chất lượng sản phẩm, phân tích ảnh giao thông…). Có nhiều họ kiến trúc: CNN cổ điển, họ YOLO, và các kiến trúc mới dựa trên Transformer/Vision Transformer — mỗi loại có ưu/nhược riêng về accuracy, tốc độ, độ phức tạp huấn luyện, khả năng triển khai thực tế.

**Yêu cầu 1 (8 điểm):**
- Chọn một **topic cụ thể** cho bài toán object detection, **tự xây dựng (construct) dataset tương ứng**.
  - Dataset phải có **tối thiểu 5 class**.
  - Số lượng mẫu tuỳ chọn nhưng phải **đủ để việc train/eval có ý nghĩa**.
- Train và đánh giá **tối thiểu 3 kiến trúc khác nhau** để so sánh, khuyến khích chọn đại diện từ 3 hướng tiếp cận:
  - **Traditional CNN-based:** Faster R-CNN, SSD, RetinaNet…
  - **YOLO-based:** các model họ YOLO.
  - **Transformer/Vision Transformer-based:** DETR, Deformable DETR, ViTDet…
- Đánh giá bằng metric phù hợp; báo cáo phải mô tả rõ cách chia train/validation/test, so sánh kết quả các model.
- Báo cáo phải nêu rõ **ưu/nhược điểm từng kiến trúc**: accuracy, tốc độ xử lý, độ phức tạp khi train, khả năng áp dụng thực tế.

**Yêu cầu 2 (2 điểm):**
- Dựa trên kết quả thực nghiệm, chọn model **tốt nhất**, xây dựng **web app**: người dùng upload ảnh → hệ thống trả về kết quả detection.

### 2. Bổ sung từ ghi chú buổi họp
- **Vai trò mô phỏng:** Research Engineer — **không cần build kiến trúc mới**, được phép dùng model có sẵn rồi **fine-tune**; không cần hiểu sâu cơ chế bên trong (YOLO hoạt động ra sao, ViT cấu trúc thế nào…) — hiểu sơ là đủ.
- **Về "construct a corresponding dataset":** mục đích thật sự là để hiểu rõ **vấn đề của dataset** (lỗi, thiếu case…) — **được phép dùng public dataset**, không bắt buộc tự gán nhãn 100%, miễn là có bước **phân tích dataset** rõ ràng và rút ra insight/hành động xử lý.
- Với các lệnh/hành động mô phỏng trong demo (không áp dụng trực tiếp cho đề này nhưng cùng tinh thần): chỉ cần chạy đúng, không cần làm gì quá cầu kỳ ở phần web.

### 3. Đánh giá & phân tích implementation

**Việc cần làm cụ thể:**
1. Chọn topic (ví dụ: phát hiện phương tiện giao thông, phát hiện lỗi sản phẩm, phát hiện đối tượng trong ảnh an ninh…) — nên chọn topic có **dataset public sẵn** để đỡ tốn thời gian gán nhãn.
2. Thu thập/chọn dataset ≥ 5 class, phân tích chất lượng dataset (class imbalance, nhãn sai, thiếu case…).
3. Chia train/val/test.
4. Chọn 1 đại diện mỗi hướng kiến trúc (ví dụ: Faster R-CNN, YOLOv8/v11, DETR/RT-DETR) → fine-tune trên dataset.
5. Đánh giá bằng mAP (mean Average Precision), IoU, tốc độ inference (FPS)…
6. Viết báo cáo so sánh 3 kiến trúc.
7. Build web app đơn giản (upload ảnh → trả bounding box).

**Ưu điểm khi triển khai:**
- Hệ sinh thái pretrained model cho object detection **rất trưởng thành và phổ biến** (Ultralytics YOLO, torchvision detection models, HuggingFace DETR…) → fine-tune nhanh, ít code từ đầu.
- Metric chuẩn hoá rõ ràng (mAP, IoU) → dễ so sánh khách quan giữa các kiến trúc.
- Không yêu cầu hiểu sâu lý thuyết → rủi ro "bị hỏi sâu" trong vấn đáp thấp hơn đề 2.

**Nhược điểm / khó khăn:**
- Cần **bounding box annotation** — nếu dataset public không sẵn nhãn đúng định dạng, tốn thời gian convert (COCO/YOLO format...) hoặc tự gán bằng tool (LabelImg, CVAT, Roboflow).
- **Train 3 kiến trúc = 3 pipeline riêng biệt** (khác input resolution, augmentation, hyperparameter, thậm chí framework khác nhau: Detectron2/torchvision cho Faster R-CNN, Ultralytics cho YOLO, HuggingFace/MMDetection cho DETR) → tốn compute và thời gian setup môi trường.
- Model dạng DETR/Vision Transformer thường **hội tụ chậm hơn, cần nhiều data hơn** để đạt hiệu năng tốt so với CNN/YOLO — nếu dataset nhỏ, đừng ngạc nhiên nếu DETR thua YOLO, đây là kết quả hợp lý cần giải thích trong báo cáo chứ không phải lỗi.
- So sánh "công bằng" giữa 3 kiến trúc khó vì mỗi framework có default khác nhau (resolution, augmentation) — cần cố gắng chuẩn hoá pipeline eval (cùng test set, cùng metric).

<!-- **Ước tính thời gian (nếu làm nghiêm túc, không rush):**
| Công việc | Thời gian |
|---|---|
| Chọn topic + chuẩn bị dataset | 1–3 ngày (public) / nhiều hơn nếu tự gán nhãn |
| Setup + train + tune 3 model | 2–4 ngày (phụ thuộc GPU/compute) |
| Web demo | 0.5–1 ngày |
| Viết báo cáo so sánh | 1 ngày |
| **Tổng** | **~1–1.5 tuần** | -->

**Vấn đề cần lưu ý:**
- Kiểm tra compute khả dụng trước (Kaggle/Colab free tier có giới hạn GPU-hour) vì phải train 3 model.
- Đảm bảo dataset đủ balance giữa các class, tránh 1 class chiếm đa số làm sai lệch mAP.
- Web demo có thể dùng Gradio/Streamlit để làm nhanh (ít điểm hơn, không cần đầu tư nhiều), hoặc FastAPI + HTML nếu muốn polish thêm cho CV.
- Nếu dùng dataset/pretrained public, nên trích dẫn nguồn trong báo cáo.

---

## ĐỀ 2 — Transformer-Based NLP Application

### 1. Yêu cầu chính thức từ đề bài

**Bối cảnh:** LLM hiện đại dựa trên kiến trúc Transformer, cơ chế self-attention giúp đạt performance tốt cả về accuracy lẫn tốc độ so với các kiến trúc trước.

**Bước 1 — chọn bài toán NLP**, các gợi ý:
- Part-of-Speech (PoS) Tagging
- Text Classification
- Named Entity Recognition (NER)
- Machine Translation
- Text Summarization
- Question Answering
- Hoặc bài toán NLP khác

**Yêu cầu 1 (8 điểm):**
- Chọn model dựa trên kiến trúc Transformer phù hợp với bài toán đã chọn.
- Tìm dataset phù hợp, **train hoặc fine-tune** model trên dataset đó.
- Đánh giá bằng metric phù hợp.
- Báo cáo mô tả chi tiết: dataset, model đã chọn, cách chia train/val/test, kết quả đánh giá.

**Yêu cầu 2 (2 điểm):**
- Build **web app** sử dụng model đã train.

### 2. Bổ sung từ ghi chú buổi họp
- **Vai trò mô phỏng:** Research Scientist — khác biệt cốt lõi so với đề 1: **bắt buộc hiểu sâu kiến trúc** model dùng (layer/block nào, loss function, cách optimize, tokenizer, và **model pretrained được train trên task gì**).
- **Chỉ cần 1 model duy nhất** (không chạy nhiều thí nghiệm như đề 1) — nhưng phải hiểu sâu.
- Nếu pipeline có hơn 1 model (ví dụ: model layout detection + model transformer xử lý layout), **chỉ cần trình bày kỹ 1 model chính** (đầu tư nhiều nhất), cái còn lại là điểm cộng, không bắt buộc trình bày sâu.
- Ví dụ minh hoạ xuyên suốt buổi học: **NER** — và toàn bộ phần lý thuyết Transformer (Encoder, Self-Attention, Positional Encoding, Multi-head, Skip Connection, Layer Norm…) trong ghi chú trước là **kiến thức nền bắt buộc** cho đề này, vì NER thuộc dạng representation learning nên **chỉ cần hiểu phần Encoder** là đủ (không cần Decoder).
- Nguyên tắc bắt buộc: phải biết pretrained model dùng được train trên (những) task gì (ví dụ BERT: MLM + NSP; PhoBERT: chỉ MLM, fine-tune tiếp từ RoBERTa).

### 3. Đánh giá & phân tích implementation

**So sánh độ khó giữa các task gợi ý** (quan trọng nhất khi chọn task cho đề này):

| Task | Độ khó | Ghi chú |
|---|---|---|
| PoS Tagging | Thấp | Nhiều dataset chuẩn, số class ít, model nhỏ đã đủ tốt |
| Text Classification | Thấp | Nhanh, dataset sẵn nhiều, dễ đạt kết quả tốt |
| NER | Trung bình | Cần hiểu BIO tagging scheme, đánh giá theo entity-level F1 (không phải token-level) |
| Machine Translation | Cao | Cần kiến trúc encoder-decoder, đánh giá BLEU, cần dataset song ngữ chất lượng, tốn compute hơn |
| Text Summarization | Cao | Seq2seq, đánh giá ROUGE, chất lượng output mang tính chủ quan, khó đánh giá tự động chính xác |
| Question Answering | Cao | Cần dataset dạng SQuAD (extractive) hoặc phức tạp hơn nếu generative |

**Ưu điểm khi triển khai:**
- Hệ sinh thái **HuggingFace** (`transformers`, `datasets`, `evaluate`) cực kỳ mạnh, hỗ trợ gần như mọi task trên — nếu đã quen, triển khai rất nhanh.
- Vì chỉ cần 1 model, không phải setup nhiều pipeline như đề 1.

**Nhược điểm / khó khăn:**
- **Rào cản lớn nhất là yêu cầu hiểu sâu** — không chỉ chạy code mà phải giải thích được cơ chế bên trong khi vấn đáp (đây là điểm khác biệt bản chất so với đề 1).
- Nếu chọn model lớn (BERT-large, T5, mBART…) để fine-tune, có thể cần GPU mạnh hơn hoặc kỹ thuật tiết kiệm tài nguyên (LoRA/PEFT).
- Các task như Machine Translation, Summarization đòi hỏi metric phức tạp hơn (BLEU, ROUGE) và khó đánh giá "đúng/sai" rạch ròi như classification.

<!-- **Ước tính thời gian:**
| Công việc | Thời gian |
|---|---|
| Chọn task + dataset | 0.5–1 ngày |
| Đọc hiểu sâu kiến trúc, tokenizer, pretrain-task của model chọn | 1–2 ngày (**đừng bỏ qua**, sẽ bị hỏi vấn đáp) |
| Fine-tune + đánh giá | 1–2 ngày |
| Web demo | 0.5 ngày |
| Viết báo cáo giải thích kiến trúc | 1 ngày |
| **Tổng** | **~1 tuần** | -->

**Vấn đề cần lưu ý:**
- Nên chọn task **cân bằng giữa "đủ thú vị để trình bày" và "đủ đơn giản để hiểu sâu trong thời gian có hạn"** — NER hoặc Text Classification là lựa chọn an toàn nhất; Machine Translation/Summarization/QA đẹp hơn cho CV nhưng rủi ro thời gian cao hơn.
- Chuẩn bị sẵn câu trả lời cho các câu hỏi sâu thường gặp: attention hoạt động cụ thể ra sao trong model đã chọn, tokenizer dùng thuật toán gì (BPE/WordPiece/SentencePiece), loss function chính xác, model pretrain trên task/dataset gì.

---

## ĐỀ 3 — Secure Virtual Assistant with Speaker Recognition

### 1. Yêu cầu chính thức từ đề bài

**Bối cảnh:** Trợ lý ảo phổ biến, hỗ trợ nhiều tác vụ (tra cứu thông tin, đặt nhắc nhở, điều khiển thiết bị, truy cập dữ liệu cá nhân, thay đổi cài đặt hệ thống). Không phải chức năng nào cũng nên thực thi ngay — tác vụ nhạy cảm cần xác thực (authenticate) trước; speaker identification dùng để cá nhân hoá trải nghiệm.

**Yêu cầu 1 (5 điểm):**
- Chọn một model đại diện cho **speaker verification hoặc speaker identification** (ví dụ **ECAPA-TDNN, RawNet3**, hoặc model tương đương).
- Dùng dataset phù hợp để **train và đánh giá**.
- Báo cáo mô tả rõ: dataset, cách chia train/val/test, model đã chọn, quy trình training, metric đánh giá phù hợp, kết quả thực nghiệm.

**Yêu cầu 2 (5 điểm):**
- Dùng model ở Yêu cầu 1, xây dựng **trợ lý ảo hoàn chỉnh** tích hợp cả speaker verification lẫn speaker identification, thoả:
  - Tương tác bằng **giọng nói (voice-based)**.
  - Có **thành phần enrollment và quản lý người dùng** (web hoặc app đơn giản) để thu mẫu giọng ban đầu và quản lý thông tin cần cho SV/SID.
  - Tối thiểu **3 loại chức năng**:
    1. **General function** — không cần xác thực.
    2. **Important function** — chỉ thực hiện được sau khi **speaker verification thành công**.
    3. **Function dùng speaker identification** — cá nhân hoá phản hồi/thông tin/cài đặt theo từng người dùng đã đăng ký.
  - Mô tả rõ **quy trình enrollment** cho user mới: cách thu mẫu giọng ban đầu, cách lưu trữ thông tin cần cho SV/SID.
  - Báo cáo phải trình bày rõ **kiến trúc hệ thống tổng thể** và **luồng xử lý (processing flow)**.

**Gợi ý kiến trúc triển khai (chính thức từ đề bài):**
- Pipeline gợi ý: **thu âm → ASR → phân tích yêu cầu & điều phối tác vụ (request analysis/task orchestration) → TTS**. Với tác vụ quan trọng, hệ thống phải có bước **SV trước khi thực thi hành động**; SID có thể được thêm vào để cá nhân hoá.
- Module phân tích yêu cầu/điều phối có thể triển khai theo 1 trong 3 hướng:
  - **Rule-based:** tập luật/pattern ánh xạ trực tiếp input → tác vụ.
  - **Intent–Entity-based:** phân loại intent + trích xuất entity, có thể dùng framework như **Rasa**.
  - **LLM-based:** dùng LLM để suy luận intent và điều phối tác vụ.

### 2. Bổ sung từ ghi chú buổi họp
- Model speaker recognition cho **tiếng Việt hiện thường chưa tốt** (thiếu dataset tiếng Việt trước đây, các năm gần đây đã khá hơn) → **khả năng cao cần fine-tune**, không nên mặc định pretrained tiếng Anh sẽ đủ tốt.
- Cần có **bước kiểm tra chất lượng mẫu enrollment giọng nói** (tương tự check chất lượng ảnh khi đăng ký khuôn mặt/vân tay) — nhưng đây là hướng nghiên cứu **chưa có nhiều tài liệu chất lượng**, đặc biệt cho tiếng Việt (low-resource language). Giảng viên có chia sẻ 2 bài báo tham khảo về cách lấy mẫu tốt (không phải về model).
- **3 use case bắt buộc demo được thật:**
  1. Chức năng thường (không xác thực) — ví dụ phát nhạc.
  2. Speaker Identification — cá nhân hoá (ví dụ gợi ý nhạc theo đặc điểm người dùng đã đăng ký).
  3. Speaker Verification — xác thực thẩm quyền cho lệnh nhạy cảm.
- Với hành động phức tạp cần phần cứng thật (mở cửa…): **chỉ cần mô phỏng** (log lại hành động). Với lệnh đơn giản làm được thật (phát nhạc…): **nên chạy thật** để demo thuyết phục hơn.
<!-- - **Kiến trúc bạn đã confirm** (khớp với 1 trong 3 hướng gợi ý chính thức — approach LLM-based):
  - Backend: **FastAPI** · Frontend: **React + Vite + TailwindCSS**
  - ASR: **Whisper** · TTS: **gTTS**
  - Speaker model: **SpeechBrain ECAPA-TDNN** (dùng inference-only trong app, train/eval độc lập trên Kaggle)
  - NLU: **LLM-based** (Anthropic/OpenAI function calling) — đúng hướng thứ 3 trong 3 hướng đề bài gợi ý
  - Database: **SQLite** (mock employee data)
  - **Ràng buộc thiết kế quan trọng đã tự đặt ra:** logic xác thực (SV/SID gating) phải **hardcode trong backend Python**, không giao cho LLM quyết định — đây chính là cách hiện thực hoá đúng tinh thần yêu cầu "hệ thống phải có bước SV trước khi thực thi hành động quan trọng" của đề bài, đồng thời tránh rủi ro bảo mật khi để LLM (vốn không đáng tin cậy tuyệt đối) tự quyết định ai được phép làm gì. -->

### 3. Đánh giá & phân tích implementation

**Ưu điểm khi triển khai:**
- **Thang điểm cân bằng nhất (5–5)** trong 3 đề → phần tích hợp ứng dụng được coi trọng ngang phần model.
- Use case rõ ràng, **demo bằng giọng nói luôn ấn tượng hơn** so với demo dạng upload ảnh/text thuần của đề 1, 2.
- **SpeechBrain ECAPA-TDNN là pretrained sẵn, chất lượng tốt**, dùng inference-only giúp tiết kiệm rất nhiều thời gian so với việc phải tự train from-scratch (nếu performance đủ tốt, không bắt buộc phải fine-tune ngay).
- **Giá trị CV cao nhất** trong 3 đề — enterprise use case.

**Nhược điểm / rủi ro:**
- **Độ phức tạp tích hợp hệ thống cao nhất** trong 3 đề: phải ráp cùng lúc ASR + NLU(LLM) + TTS + SV/SID + Database + Web frontend — nhiều điểm có thể phát sinh lỗi (audio pipeline, latency, quyền truy cập micro trên browser, định dạng/sample rate audio…).
- **Audio pipeline** (ghi âm từ browser → gửi backend → format/sample rate đúng chuẩn cho Whisper và ECAPA-TDNN → streaming hay batch) thường là **nguồn phát sinh bug/khó debug nhất** trong dạng dự án này.
- Model SV/SID cho tiếng Việt có thể chưa đủ tốt → nếu cần fine-tune, phải có dataset giọng nói tiếng Việt (có thể cân nhắc VLSP, VIVOS, Common Voice vi, hoặc tự thu âm) — và **chất lượng mẫu enrollment kém sẽ ảnh hưởng trực tiếp đến accuracy toàn hệ thống**, nên bước enrollment quality check không nên bỏ qua dù đơn giản.
- **Chọn threshold cho speaker verification** đòi hỏi hiểu trade-off giữa bảo mật và trải nghiệm người dùng (False Accept Rate vs False Reject Rate, thường dùng EER — Equal Error Rate làm điểm tham chiếu) — cần có cơ sở thực nghiệm, không nên chọn số tuỳ ý rồi giải thích qua loa.
- **Giới hạn bảo mật cố hữu:** chỉ xác thực bằng giọng nói có thể bị tấn công bằng bản ghi âm phát lại (replay attack) hoặc deepfake giọng nói — đề bài không bắt buộc phòng chống, nhưng nên **nêu rõ giới hạn này trong báo cáo** vì giám khảo có thể hỏi tới.
- Vì dùng **LLM cho NLU**, mỗi lệnh voice đều cần gọi API bên ngoài → **latency** có thể cao hơn so với rule-based/RASA thuần — cần cân nhắc trải nghiệm người dùng (voice assistant mà phản hồi chậm sẽ giảm điểm demo).
<!-- 
**Ước tính thời gian** (dựa trên kế hoạch hiện tại: build app trong 1–2 ngày bằng Claude Code Pro, train model làm riêng trên Kaggle):
| Công việc | Thời gian | Ghi chú |
|---|---|---|
| Train/eval SV-SID model (Kaggle) | Độc lập, không tính vào 1–2 ngày build app | Bạn đang tự làm phần này |
| Build app 8 module (M0–M7) | **1–2 ngày** theo kế hoạch | Khả thi **nếu**: pretrained ECAPA-TDNN đủ tốt (không cần fine-tune phức tạp ngay trong lúc build app), spec module đã sẵn chi tiết (đã có), audio pipeline không phát sinh vấn đề lớn |
| Buffer rủi ro audio pipeline | +0.5–1 ngày | Nên dự trù, đây là điểm dễ trễ tiến độ nhất |
| Viết báo cáo (train SV riêng + phần tích hợp) | ~1 ngày | Gồm cả phần giải thích architecture & processing flow theo đúng yêu cầu đề bài | -->

**Vấn đề cần lưu ý đặc biệt (ưu tiên xử lý sớm):**
1. **Test riêng ASR (Whisper) và TTS (gTTS) trước khi build UI phức tạp** — front-load rủi ro lớn nhất của cả đồ án.
2. **Enrollment quality check**: nên có dù đơn giản (kiểm tra độ dài audio tối thiểu, số lượng câu enroll, SNR cơ bản…) vì đề bài yêu cầu mô tả rõ enrollment procedure trong báo cáo.
3. **Threshold verification** cần chọn dựa trên EER đo được trên tập validation, không hardcode tuỳ ý.
4. **Latency của LLM-based NLU**: cân nhắc dùng model nhanh/rẻ cho bước intent classification, hoặc thêm caching nếu cần, để trải nghiệm voice assistant không bị "đơ" khi demo.
5. Vì logic auth hardcode ở backend (đúng hướng), hãy đảm bảo báo cáo **giải thích rõ vì sao** thiết kế này an toàn hơn để LLM tự quyết định — đây là điểm cộng về mặt tư duy hệ thống khi vấn đáp.

---

## Bảng so sánh tổng hợp 3 đề

| Tiêu chí | Đề 1: Object Detection | Đề 2: NLP Transformer | Đề 3: Voice Assistant |
|---|---|---|---|
| Điểm phân bổ | 8 (model) + 2 (web) | 8 (model) + 2 (web) | 5 (model) + 5 (app) |
| Số model cần train | ≥ 3 kiến trúc | 1 model | 1 model |
| Độ sâu lý thuyết yêu cầu | Thấp (Research Engineer) | **Cao** (Research Scientist, bắt buộc hiểu sâu) | Trung bình |
| Độ phức tạp tích hợp hệ thống | Thấp (chỉ upload ảnh → kết quả) | Thấp (chỉ nhập text → kết quả) | **Cao** (audio real-time, nhiều thành phần) |
| Rủi ro lớn nhất | Compute để train 3 model, so sánh công bằng | Không hiểu đủ sâu để trả lời vấn đáp | Audio pipeline + timeline gấp |
| Giá trị CV | Trung bình–cao | Trung bình | **Cao** (enterprise use case cụ thể) |
| Ước tính thời gian nghiêm túc | ~1–1.5 tuần | ~1 tuần | ~1 tuần build app (train model riêng) |

---
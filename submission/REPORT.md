# Lab 21 — Bằng chứng và quyết định triển khai

Hoàng Công Minh · MSSV 2A202602774 · Phiên chạy 07–08/10/2026 (Asia/Bangkok).
Thực hiện với AI assistant; các con số dưới đây lấy từ artefact chạy thật.

## Thiết kế và môi trường

Model: `Qwen/Qwen3.5-0.8B`; tier `LAPTOP`. Máy thực tế là
NVIDIA GeForce RTX 3050 Ti Laptop GPU 4.0 GB, Windows, Python 3.13.15.
Precision bf16 phần cứng: True. Model 0.8B được chọn vì model 4B mặc định
không vừa GPU này; đây là thay đổi được README cho phép qua BASE_MODEL.
Revision model đã chạy: `2fc06364715b967f1860aea9cf38778875588b17`.
Giữ cấu hình tier LAPTOP: batch mỗi thiết bị 1, gradient accumulation 8,
batch hiệu dụng 8, seed 42. Corpus chính là 250 ticket CSKH tiếng Việt → JSON
intent/urgency/product/sentiment, split 225 train và 25 validation. Chọn corpus này
để phép chấm khách quan và so sánh lại được với hướng dẫn, không cần LLM judge.
Validation được tách riêng nhưng không dùng để lựa chọn checkpoint; tất cả run
được chấm ở cuối cùng ngân sách, không tìm checkpoint tốt nhất trên eval.

Độ dài đo được: p95=98, p99=100, max=101;
khuyến nghị làm tròn là 256. `max_length=1024`
giữ nguyên tier như hướng dẫn đổi base. Dataset pre-tokenized không pad đến giới hạn
này, batch=1 nên không trả chi phí padding tới 1024; không mẫu nào bị cắt.
Epochs=2; ngân sách của mỗi run là 58 optimizer steps.
Phiên bản cài thực tế được lưu trong `submission/requirements-lock.txt`.

## Chứng minh mask và template

`answer_is_supervised=True`;
`question_is_masked=True`.
Mask `assistant-only` tính loss trên 37/94
token của mẫu kiểm chứng, supervised_fraction=0.3936.

```text
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>

```

Template giữ reasoning: `True`; body hiện diện:
`True`. Nhưng template không có marker generation để API
assistant mask tự suy ra loss span. Kiểm tra tokenizer trả mask rỗng; đây là cảnh báo
về đường API đó, không phải mask đã chứng minh của labkit bị sai. Huấn luyện dùng
input_ids và labels từ build_example, có EOS được giám sát, packing tắt. Không dùng
assistant_only_loss để tránh thay template hoặc học trên mask khác với NB1.

## Mốc đóng băng trước huấn luyện

NB2 chạy trước NB3. Không thay OPTIMIZED_PROMPT hay hai tập eval sau khi thấy kết quả
huấn luyện. SHA prompt (b): `719e74d3b6232053`. Chấm đầy đủ
50 target và 15 regression; smoke_mode:
`False`. Dự đoán gốc được lưu trong baseline_predictions.json.
Điểm (b) cao hơn
(a); không làm yếu baseline để nâng thành tích fine-tune.
Checksum dữ liệu gốc khớp Git HEAD và checksums.json khi dùng LF chuẩn. Working tree
Windows dùng CRLF nên hash byte thô ban đầu cảnh báo khác; verifier chỉ chuẩn hóa
ký tự xuống dòng, vẫn phát hiện thay nhãn hoặc nội dung. Bằng chứng từng file nằm
trong data_integrity_audit.json. Không đổi nội dung train/eval của thí nghiệm chính.

| run | target | regression | format | latency_ms | n |
|---|---|---|---|---|---|
| (a) base + naive prompt | 0.0 | 0.6778 | 0.0 | 2533.1 | 50 |
| (b) base + optimized prompt | 0.49 | 0.6778 | 1.0 | 635.1 | 50 |
| (c) LoRA fine-tune | 0.985 | 0.0333 | 1.0 | 1143.4 | 50 |

Format ở đây là tỷ lệ khóa cần thiết mà parser của lab tìm được, cho phép trích JSON
trong prose/fence; không phải tỷ lệ chỉ xuất một JSON sạch. Target là trung bình tỷ lệ
đúng của bốn trường. Regression là keyword recall trên 15 câu hỏi; nó chỉ là phép thử
hồi quy nhỏ, không bao phủ toàn bộ năng lực tổng quát. Latency là trung bình mỗi mẫu
với greedy decode, batch inference 4, cùng máy; chưa phải p95 khi phục vụ một request.

## Bốn cấu hình, một ngân sách step

| run | target | r | trainable_params | learning_rate | final_loss | max_steps | train_seconds | peak_vram_gb |
|---|---|---|---|---|---|---|---|---|
| correct | 0.985 | 16 | 10822656 | 0.0001 | 0.3924 | 58 | 315.3 | 1.97 |
| attn_only | 0.915 | 271 | 10822656 | 0.0001 | 0.4335 | 58 | 221.9 | 1.98 |
| wrong_lr | 0.335 | 16 | 10822656 | 1e-05 | 1.5398 | 58 | 326.7 | 1.98 |
| qlora | 0.79 | 16 | 10822656 | 0.0001 | 0.423 | 58 | 348.2 | 1.19 |

### Vị trí so với rank

Run attn_only chỉ đổi vị trí gắn vào q/v; rank được nâng tới 271 để khớp
ngân sách 10822656 so với 10822656 của correct.
Độ lệch ngân sách là 0.0000%.
Target attn_only=0.915, correct=0.985; chênh lệch
-0.0700. Đây là phép đo ảnh hưởng vị trí trong
điều kiện đã khống chế tổng tham số, không phải bằng chứng rằng rank cao luôn tốt.
Loss tương ứng là 0.4335 và 0.3924.
Thứ tự target giảm dần: correct → attn_only → qlora → wrong_lr (đọc giá trị trong bảng để nhận biết hòa).
Thứ tự loss tăng dần: correct → qlora → attn_only → wrong_lr.
Hai thứ tự khác nhau;
chỉ số huấn luyện không đủ để kết luận tổng quát hóa.
Không suy rộng kết luận từ một model, một seed và corpus nhỏ sang mọi kiến trúc.
Kiến trúc gồm 18 lớp linear attention và 6 lớp full attention (architecture.json).
q_proj/v_proj thuộc các lớp full attention; text-linear còn gắn vào projection của
linear attention và MLP. Tăng rank q/v không thay được phạm vi lớp được tác động.
Lưu ý: cột final_loss trong source là `Trainer.training_loss` trung bình cả run,
không phải loss duy nhất của step cuối. Đường loss theo step mới dùng để chẩn đoán.

### Thang learning rate

Wrong_lr giữ r=16, vị trí text-linear, cùng dữ liệu và step; chỉ giảm LR từ
0.0001 xuống 1e-05. Loss được lưu theo step
trong correct_training_log.json và wrong_lr_training_log.json để phân biệt giảm chậm,
dao động và thực sự không học. Loss log đầu/cuối của correct là
2.8096/0.0080; wrong_lr là
3.0269/0.7743. Loss trung bình là 0.3924 so với
1.5398, target là 0.985 so với 0.335.
Không thể quy mọi chênh lệch loss cho năng lực adapter: ở đây thang cập nhật khác nhau.
Phải nhìn target và regression cùng với đường loss trước khi chọn một cấu hình.

### Chi phí lượng tử hóa

QLoRA chỉ đổi base sang NF4 4-bit; cùng vị trí, r, LR, dữ liệu và step.
Peak VRAM: 1.97 GB so với 1.19 GB,
tiết kiệm tuyệt đối 0.7800 GB. Target thay đổi
-0.1950; thời gian chạy và loss có trong bảng.
Adapter QLoRA được đánh giá trên base 4-bit đúng với khi train. Số đo này chỉ kiểm tra
khuyến nghị về lượng tử hóa trên model 0.8B của phiên chạy, không chứng minh kết quả
trên Qwen3.5-4B/T4. VRAM là max_memory_allocated của PyTorch, khác tổng bộ nhớ nvidia-smi.

## Phán quyết và ý nghĩa

**FAILED**. Target Δ=+0.4950;
regression Δ=-0.6444; valid_trace_rate của phép đo core
(thinking tắt)=0.0. Các lý do do gate ghi:

- general capability regressed by 0.644 (tolerance 0.020). See deck §6.3 — add 1-5% replay data.

Phán quyết xuất phát từ phép so sánh với baseline đã được prompt đầy đủ, không phải
từ việc loss huấn luyện giảm. Hai điều kiện của gate là target phải tăng nghiêm ngặt
và regression không giảm quá 0.02. Một bản fine-tune có thể học tốt schema mà vẫn mất
khả năng trả lời câu hỏi ngoài miền. Ngược lại, model có thể giữ regression nhưng
không vượt được prompt tối ưu; khi đó chi phí huấn luyện không được biện minh bằng
độ chính xác. Phép đo format và latency giúp hiểu giá của hành vi mới, nhưng gate
hiện tại chỉ kiểm target/regression theo đúng source. Không thay ngưỡng khi kết quả
không như kỳ vọng. Kết quả trên tập nhỏ chỉ hỗ trợ quyết định trong lab; chưa có
bootstrap confidence interval hoặc lặp nhiều seed, nên chênh lệch nhỏ chưa đủ để
khẳng định ưu thế ổn định trên traffic thật.
Các câu hỏi hồi quy cho thấy adapter áp đặt kiểu trả lời JSON triage lên cả yêu cầu
ngoài miền. Đây là suy giảm hành vi đầu ra đo được, chưa chứng minh tri thức bên
trong đã bị xóa không thể phục hồi. Có thể giữ base cho câu hỏi phổ thông và chỉ
bật adapter ở route triage, nhưng vẫn cần kiểm thử routing và ticket thật trước
khi triển khai. Không dùng kết quả merge giữ nguyên target để bỏ qua gate hồi quy.

## Định tính: so từng ca với baseline (b)

| i | nhóm | ticket | nhãn | baseline_b | fine_tune | b_score | ft_score | kết_quả |
|---|---|---|---|---|---|---|---|---|
| reg-2 | regression | 1 km bằng bao nhiêu mét? | {"keywords": ["1000"]} | 1 km (kilometr) là đơn vị đo lường dài rộng, trong đó ký hiệu **k** được viết tắt từ chữ **kilo**.  Để chuyển đổi từ km sang mét, ta cần nhớ quy tắc đơn vị: *   **kilo** = 1000 *   **m** = 100  Do đó, để chuyển đổi 1 km thành mét, ta nhân số đó bằng 1000: | {"intent": "conversion", "urgency": "low", "product": "tỷ lệ chuyển đổi"} | 1.0 | 0.0 | loss |
| reg-3 | regression | Viết một câu chúc mừng sinh nhật bằng tiếng Việt. | {"keywords": ["sinh nhật"]} | Chúc mừng sinh nhật! 🎉 Chúc bạn luôn tràn đầy năng lượng, hạnh phúc và những kỷ niệm đẹp nhất trong đời. | {"intent": "hoi_thong_tin", "urgency": "thap", "sentiment": "tieu_cuc"} | 1.0 | 0.0 | loss |
| 49 | target | Chào shop, mình đặt ốp lưng điện thoại mã đơn VN833689. Sai màu. Sớm nhé. Shop xem giúp. | {"intent": "san_pham_loi", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "trung_tinh"} | {"intent": "van_chuyen", "urgency": "cao", "product": " ốp lưng điện thoại", "sentiment": "tich_cuc"} | {"intent": "san_pham_loi", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "trung_tinh"} | 0.25 | 1.0 | win |
| 43 | target | Xin chào, mình đặt bình giữ nhiệt mã đơn DH866575. Khi nào có tiền về. Sớm nhé. Quá tệ. | {"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tieu_cuc"} | {"intent": "van_chuyen", "urgency": "cao", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"} | {"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tieu_cuc"} | 0.25 | 1.0 | win |
| 9 | target | Cho mình hỏi, mình đặt chuột không dây mã đơn OD538419. Hoàn tiền. Mong shop phản hồi. Mình vẫn tin tưởng shop. | {"intent": "hoan_tien", "urgency": "trung_binh", "product": "chuột không dây", "sentiment": "tich_cuc"} | {"intent": "hoan_tien", "urgency": "cao", "product": "chữ ký không dây", "sentiment": "tich_cuc"} | {"intent": "hoan_tien", "urgency": "thap", "product": "chuột không dây", "sentiment": "tich_cuc"} | 0.5 | 0.75 | win |

Trên target có 0 ca fine-tune thua baseline.
Tổng trên cả target và regression: thua 11;
thắng: 48;
hòa: 6.
Đã đưa ít nhất hai ca thua vào bảng.
Toàn bộ dự đoán, nhãn và score của từng mẫu được lưu không cắt ngắn trong qualitative.json.
Các ca thua ngoài miền nằm trong qualitative_regression.json và được đánh dấu nhóm
regression trong bảng; đó là câu hỏi kiến thức/chỉ dẫn, không phải ticket target.
Ví dụ của nhóm này dùng keyword recall, khác field accuracy của nhóm target.
Keyword recall có giới hạn: ở câu đổi 1 km, baseline chứa 1000 nên được điểm, nhưng
còn kèm diễn giải sai “m = 100”. Không coi điểm keyword cao là bằng chứng mọi câu
văn đều đúng; vẫn giữ nguyên output để người đọc kiểm tra giới hạn của phép chấm.
Số trường sai trên target (kể cả khi vẫn thắng baseline): {"urgency": 3}.
Các số này mô tả lỗi quan sát được, không chứng minh nguyên nhân bằng quan sát đơn lẻ.

## Kết luận và điều rút ra

Quyết định triển khai phải dựa vào baseline tốt nhất đã đo, vì người dùng có thể đạt
hành vi mong muốn bằng một prompt mà không cần bảo trì trọng số mới. Trong phiên
này, gate không đạt và đó là kết luận về phép đo hiện có,
không phải một đánh giá chung rằng LoRA tốt hoặc xấu. Nếu không vượt baseline,
không nên deploy adapter chỉ vì train loss đẹp; nếu vượt, vẫn cần thử ticket thật,
schema nghiêm ngặt và các câu hỏi ngoài miền phong phú hơn trước khi phát hành.
Mask đúng là điều kiện nền tảng: khi câu hỏi cũng bị tính loss, model tối ưu một tác
vụ khác và mọi so sánh sau đó mất ý nghĩa. LR, vị trí và rank chỉ có thể được phân
tích sau khi template và prompt huấn luyện khớp với prompt suy luận. Việc khống chế
ngân sách tham số tránh gán nhầm tác dụng của số tham số cho vị trí adapter; khống
chế số step tránh gán tác dụng của thời gian học cho lượng tử hóa hoặc learning rate.
Dữ liệu seed gồm nhiều cách diễn đạt cùng nhãn hẹp, nên khả năng ghi nhớ là một lời
giải thích cần kiểm tra, không thể loại bỏ chỉ bằng loss thấp. Tôi sẽ ưu tiên mở rộng
kiểm thử với ca mơ hồ, thực thể chưa thấy và ticket của nguồn khác, rồi lặp nhiều seed
trước khi tốn thêm GPU cho rank lớn. Nếu có thêm hai giờ, tôi sẽ đo khoảng tin cậy
cho target và kiểm thử hồi quy đa miền thay vì chỉnh gate để biến FAIL thành PASS.

Ba điều rút ra từ thao tác cụ thể: template giữ reasoning không đồng nghĩa API mask
tự hoạt động; ca fine-tune có score thấp chưa chắc là ca thua baseline; checkpoint
an toàn phải được lưu trước evaluation vì inference cũng có thể OOM. Những điều này
được chứng minh bằng artefact và luồng thực thi, không chỉ là khuyến nghị chung.

## Phụ lục bonus

| run | r | lora_alpha | trainable_params | learning_rate | max_steps | mask_mode | optimizer | train_seconds | peak_vram_gb |
|---|---|---|---|---|---|---|---|---|---|
| rank_8 | 8 | 16 | 5411328 | 0.0001 | 58 | assistant-only | adamw | 330.2 | 1.88 |
| rank_64 | 64 | 128 | 43290624 | 0.0001 | 58 | assistant-only | adamw | 335.0 | 2.53 |
| trace_assistant_only | 16 | 32 | 10822656 | 0.0001 | 58 | assistant-only | adamw | 392.6 | 2.15 |
| trace_response_only | 16 | 32 | 10822656 | 0.0001 | 58 | response-only | adamw | 382.0 | 2.15 |
| lab_support | 16 | 32 | 10822656 | 0.0001 | 86 | assistant-only | adamw | 531.4 | 2.01 |
| muon | 16 | 32 | 10822656 | 0.001 | 58 | assistant-only | muon | 375.1 | 1.93 |

### B1 — Merge

```json
{
  "before_merge": 0.985,
  "after_merge": 0.985,
  "delta": 0.0,
  "tolerance": 0.01,
  "n": 50
}
```

### B1 — Hot-swap

```json
{
  "adapters": [
    "correct",
    "attn_only",
    "wrong_lr"
  ],
  "ticket": "Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt.",
  "predictions": [
    {
      "adapter": "correct",
      "prediction": "{\"intent\": \"doi_tra\", \"urgency\": \"cao\", \"product\": \"chuột không dây\", \"sentiment\": \"tich_cuc\"}"
    },
    {
      "adapter": "attn_only",
      "prediction": "{\"intent\": \"doi_tra\", \"urgency\": \"cao\", \"product\": \"chuột không dây\", \"sentiment\": \"tich_cuc\"}"
    },
    {
      "adapter": "wrong_lr",
      "prediction": "{\"intent\": \"sua\", \"urgency\": \"cao\", \"product\": \"chuột không dây\", \"sentiment\": \"cảm ơn\"}"
    }
  ],
  "same_base": true
}
```

### B2 — Baseline miền phòng lab

```json
{
  "baseline_a": {
    "run": "base",
    "thinking": false,
    "n": 75,
    "max_new_tokens": 160,
    "target": 0.0,
    "regression": 0.6777777777777777,
    "format": 0.0,
    "latency_ms": 2741.0383586666044,
    "valid_trace_rate": 0.0
  },
  "baseline_b": {
    "run": "base",
    "thinking": false,
    "n": 75,
    "max_new_tokens": 160,
    "target": 0.3433333333333333,
    "regression": 0.6777777777777777,
    "format": 1.0,
    "latency_ms": 993.9502253333922,
    "valid_trace_rate": 0.0
  },
  "optimized_prompt": "Bạn phân loại yêu cầu hỗ trợ phòng lab máy tính. Chỉ trả về JSON với 4 khóa intent, urgency, product, sentiment. intent thuộc mat_ket_noi, loi_phan_mem, thiet_bi_loi, tai_khoan, yeu_cau_cai_dat. urgency: cao nếu thi/kiểm tra đang diễn ra; trung_binh nếu đang thực hành; thap nếu chuẩn bị buổi sau. sentiment: tieu_cuc nếu bực mình; trung_tinh nếu thông báo; tich_cuc nếu cảm ơn và hài lòng. product là tên máy nguyên văn, không bao gồm mã thiết bị.",
  "eval_sha256": "086e73b1d5ca07e674b38f4960859bc4e21c9960199f00122917e66d02917469",
  "model": "Qwen/Qwen3.5-0.8B"
}
```

### B2 — Kết quả miền phòng lab

```json
{
  "scores": {
    "run": "lab_support",
    "thinking": false,
    "n": 75,
    "max_new_tokens": 160,
    "target": 0.9866666666666667,
    "regression": 0.13333333333333333,
    "format": 1.0,
    "latency_ms": 983.0935293331762,
    "valid_trace_rate": 0.0
  },
  "verdict": {
    "passed": false,
    "reasons": [
      "general capability regressed by 0.544 (tolerance 0.020). See deck §6.3 — add 1-5% replay data."
    ],
    "target_delta": 0.6433333333333333,
    "regression_delta": -0.5444444444444444
  },
  "n_train": 337,
  "n_val": 38
}
```

### B3 — Mask có khác nhau thật không

```json
[
  {
    "mask_mode": "assistant-only",
    "n": 225,
    "supervised_tokens": 19124,
    "preview": "Ticket nhắc đến tai nghe bluetooth. Nội dung yêu cầu thuộc nhóm san_pham_loi; tín hiệu thời gian và thái độ được gán lần lượt trung_binh và tieu_cuc theo nhãn của bộ dữ liệu.\n</think>\n\n{\"intent\": \"san_pham_loi\", \"urgency\": \"trung_binh\", \"product\": \"tai nghe bluetooth\", \"sentiment\": \"tieu_cuc\"}<|im_end|>\n"
  },
  {
    "mask_mode": "response-only",
    "n": 225,
    "supervised_tokens": 8789,
    "preview": "\n\n{\"intent\": \"san_pham_loi\", \"urgency\": \"trung_binh\", \"product\": \"tai nghe bluetooth\", \"sentiment\": \"tieu_cuc\"}<|im_end|>\n"
  }
]
```

### B3 — Token đầu vào giống nhau

```json
{
  "input_sha256_by_mask": {
    "assistant-only": "82a188084b1099795468bb591c9cadfde47806d5a7f06a4a14e1963d3c3e5414",
    "response-only": "82a188084b1099795468bb591c9cadfde47806d5a7f06a4a14e1963d3c3e5414"
  },
  "identical_inputs": true,
  "token_stats": {
    "n": 225,
    "mean": 138.0,
    "p50": 138,
    "p95": 145,
    "p99": 148,
    "max": 149,
    "suggested_max_length": 256
  }
}
```

### B3 — Thinking bật

```json
[
  {
    "run": "base",
    "thinking": true,
    "n": 50,
    "max_new_tokens": 512,
    "target": 0.0,
    "regression": 0.6777777777777777,
    "format": 0.0,
    "latency_ms": 8612.637253999928,
    "valid_trace_rate": 0.0
  },
  {
    "run": "trace_assistant_only",
    "thinking": true,
    "n": 50,
    "max_new_tokens": 512,
    "target": 0.915,
    "regression": 0.5333333333333333,
    "format": 1.0,
    "latency_ms": 2600.291283999977,
    "valid_trace_rate": 1.0,
    "mask_mode": "assistant-only"
  },
  {
    "run": "trace_response_only",
    "thinking": true,
    "n": 50,
    "max_new_tokens": 512,
    "target": 0.095,
    "regression": 0.23333333333333334,
    "format": 0.44,
    "latency_ms": 4165.943574000121,
    "valid_trace_rate": 0.02,
    "mask_mode": "response-only"
  }
]
```

### B4 — Rank cố định vị trí

```json
[
  {
    "run": "correct",
    "thinking": false,
    "n": 50,
    "max_new_tokens": 160,
    "target": 0.985,
    "regression": 0.03333333333333333,
    "format": 1.0,
    "latency_ms": 1147.1346060000724,
    "valid_trace_rate": 0.0,
    "rank": 16,
    "target_modules": "text-linear"
  },
  {
    "run": "rank_8",
    "thinking": false,
    "n": 50,
    "max_new_tokens": 160,
    "target": 0.93,
    "regression": 0.13333333333333333,
    "format": 1.0,
    "latency_ms": 1149.4866999999795,
    "valid_trace_rate": 0.0,
    "rank": 8,
    "target_modules": "text-linear"
  },
  {
    "run": "rank_64",
    "thinking": false,
    "n": 50,
    "max_new_tokens": 160,
    "target": 1.0,
    "regression": 0.0,
    "format": 1.0,
    "latency_ms": 1101.2055159998636,
    "valid_trace_rate": 0.0,
    "rank": 64,
    "target_modules": "text-linear"
  }
]
```

### B6 — Dự đoán trước khi chạy

```json
{
  "hypothesis": "Muon may underperform AdamW on target at this small step budget; LoRA may limit damage.",
  "lr": 0.001,
  "momentum": 0.95,
  "adjust_lr_fn": "match_rms_adamw",
  "reason": "Use Muon's own default LR and matrix RMS scaling; do not copy AdamW's LR.",
  "limitation": "Base pretraining optimizer is unverified; this does not isolate optimizer mismatch."
}
```

### B6 — Optimizer Muon

```json
{
  "run": "muon",
  "thinking": false,
  "n": 50,
  "max_new_tokens": 160,
  "target": 0.0,
  "regression": 0.0,
  "format": 0.0,
  "latency_ms": 4479.589313999968,
  "valid_trace_rate": 0.0
}
```

### B6 — Chẩn đoán output (bổ sung)

```json
{
  "diagnostic_only": true,
  "official_scores_unchanged": true,
  "n": 50,
  "n_first_object_parseable": 50,
  "n_extra_text_after_first_object": 50,
  "first_object_field_accuracy": 1.0,
  "interpretation": "Supplementary parser diagnostic only; repeated JSON breaks official scorer. Do not substitute for official scores."
}
```

B2 huấn luyện 337 mẫu, giữ lại 38 validation; 75 mẫu eval có cách diễn đạt và mã máy riêng. Target so với baseline tối ưu đổi +0.6433; regression đổi -0.5444; gate FAILED. Không đổi ngưỡng gate cho miền riêng. Corpus này mới về cách kết hợp nhãn và bối cảnh phòng học so với corpus CSKH của lab, nhưng không chứng minh base chưa gặp khái niệm kỹ thuật khi pretrain. Kết quả chỉ kiểm tra chuyển sang schema nghiệp vụ trong miền tổng hợp hẹp.

Biên độ ảnh hưởng trên target: {"rank": 0.06999999999999995, "vị trí": 0.06999999999999995, "LR": 0.6499999999999999}.

Thứ tự theo biên độ: LR > rank = vị trí.

Tăng r=16 lên r=64 dùng gấp 4 ngân sách tham số LoRA, target tăng +0.0150, regression đổi -0.0333. r=16 đã gần trần target của tập này; lợi ích rank lớn phải được cân với bộ nhớ, khả năng ngoài miền và đánh giá trên dữ liệu mới.

Corpus nhỏ có thể chưa dùng hết năng lực r=64; phải nhìn điểm giữ lại thay vì giả định nhiều tham số thì tốt hơn. Không có learning curve theo kích thước dữ liệu nên chưa thể xác định ngưỡng thông tin mà rank lớn trở nên cần thiết.

| MASK_MODE | target | valid_trace_rate | regression |
|---|---|---|---|
| base (chưa fine-tune) | 0.0 | 0.0 | 0.6777777777777777 |
| assistant-only | 0.915 | 1.0 | 0.5333333333333333 |
| response-only | 0.095 | 0.02 | 0.23333333333333334 |

Base thinking có valid_trace_rate=0 trong ngân sách 512 token; không có mốc trace hợp lệ cao trước train để chứng minh suy giảm so với base. Output gốc được giữ trong base_bonus_predictions.json để kiểm tra lặp/cắt ngắn. So sánh hai mask vẫn đo hành vi trace, nhưng không đủ để tuyên bố tái lập mất năng lực suy luận từ base.

B3 response-only so với assistant-only: target Δ=-0.8200, trace Δ=-0.9800. Nhìn target một mình không đo được hình thức reasoning. Phiên này không quan sát thấy cặp dấu target tăng/trace giảm giữa hai mask. Đây là kết quả cần báo cáo nguyên trạng.

Ở response-only, trace vẫn nằm trong context khi train (teacher forcing), nhưng token của trace không được giám sát. Khi eval thinking bật, model phải tự sinh phần context đó trước JSON; loss JSON thấp trên context có sẵn không bảo đảm sinh được toàn chuỗi. Đây là một cơ chế khả dĩ cho chênh lệch đã đo, chưa được tách riêng bằng thí nghiệm can thiệp. Không so trực tiếp loss giữa hai mask như cùng một mục tiêu vì số token được giám sát khác nhau.

B6 target Muon Δ=-0.9850 so với AdamW. Giả thuyết Muon có thể kém hơn được viết trước khi train; không thay đổi dự đoán sau khi thấy điểm. Vì pretraining optimizer chưa được xác minh và LR cũng đổi theo họ optimizer, thí nghiệm này không tách riêng tác động optimizer mismatch.

Dự đoán target thấp hơn AdamW phù hợp với kết quả quan sát. Ngân sách step nhỏ, các ma trận adapter và lựa chọn LR đều có thể ảnh hưởng; không có ablation đủ để gán nguyên nhân cho một yếu tố.

B6 có 50/50 output đọc được object đầu và 50 output còn nội dung phía sau; output lặp JSON tới giới hạn token. Chỉ chấm object đầu cho field accuracy=1.0000, nhưng đây là chẩn đoán bổ sung, không thay phép chấm chính. Target/format chính vẫn bằng 0 vì parser không nhận chuỗi nhiều JSON. Do đó số 0 không chứng minh model không biết phân loại; lỗi hành vi kết thúc output làm hỏng giao thức trả lời. Giữ output gốc và scorer chính để phép so sánh không bị chỉnh sau kết quả.

Merge bỏ overhead adapter trong đồ thị suy luận nhưng làm mất khả năng chọn
adapter riêng ngay trên cùng base chưa merge; giữ adapter riêng phù hợp với nhiều tác
vụ/khách hàng hoặc cần rollback nhanh. B2 là corpus tổng hợp có tuyên bố nguồn và hạn
chế tại data/CUSTOM_DATASET.md, không giả là ticket thật. B3 dùng cùng corpus có trace
tổng hợp từ nhãn, hai mask và bật thinking khi eval; so với base chưa train. Trace mở
được phục hồi từ generation prompt Qwen để tránh bỏ sót thẻ <think> ngoài new tokens.
Valid trace chỉ là block đóng, không rỗng, không chứng minh lập luận đúng. Các mẫu
chạm ngân sách 512 token có thể bị tính trace invalid vì cắt ngắn, không chỉ vì collapse.
Không gọi khác biệt giữa hai run là tái lập collapse nếu số đo không hỗ trợ.
B4 giữ text-linear, LR và step, alpha/r=2; r=16 dùng adapter correct của core.
B5 chưa công khai: cần tài khoản/repo và quyết định công khai của người học.
B6/B7 là thử thách không tính điểm. B6 dùng Muon có sẵn trong PyTorch của phiên chạy,
chỉ tối ưu các ma trận LoRA 2 chiều; LR theo mặc định Muon và scaling riêng, không
bê nguyên LR AdamW sang. B7 yêu cầu GPU đủ lớn cho base MoE, nên điều kiện phần cứng
không thỏa trên máy 4 GB; chưa chạy và không suy diễn kết quả.

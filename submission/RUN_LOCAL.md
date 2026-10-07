# Chạy lại trên Windows

Các lệnh dưới đây chạy từ thư mục gốc repository. Cần Python 3.13 và NVIDIA GPU.
Cấu hình phiên chạy: RTX 3050 Ti Laptop 4 GB, Qwen/Qwen3.5-0.8B, tier LAPTOP.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt bitsandbytes
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTEST_ADDOPTS = '--basetemp=.pytest_tmp'
```

Tạo `.env` với nội dung sau; không đưa token vào source hoặc file ZIP nộp bài:

```dotenv
COMPUTE_TIER=LAPTOP
BASE_MODEL=Qwen/Qwen3.5-0.8B
MASK_MODE=assistant-only
EPOCHS=2
```

Chạy lần lượt; chỉ huấn luyện sau khi đọc và đóng băng kết quả NB2:

```powershell
.\.venv\Scripts\python.exe scripts/verify.py --smoke
.\.venv\Scripts\python.exe scripts/colab_run.py nb1 nb2
.\.venv\Scripts\python.exe scripts/colab_run.py nb3 nb4 nb5 nb6
.\.venv\Scripts\python.exe scripts/run_bonus.py rank
.\.venv\Scripts\python.exe scripts/run_bonus.py trace
.\.venv\Scripts\python.exe scripts/run_bonus.py custom
.\.venv\Scripts\python.exe scripts/run_bonus.py muon
.\.venv\Scripts\python.exe scripts/write_report.py
.\.venv\Scripts\python.exe scripts/package_submission.py
```

`requirements-lock.txt` lưu đúng phiên bản của lần chạy đã báo cáo. Wheel torch CUDA
cần index riêng như lệnh trên. Corpus bonus và adapter bonus được tách riêng; chạy
bonus không ghi đè adapters/correct hoặc verdict.json của core. B3 dùng trace tổng
hợp để hai mask thực sự khác nhau, không chỉ đổi biến môi trường trên câu trả lời
JSON vốn làm hai mode tương đương. B2 kiểm tra tập train/eval không trùng input, nhưng
đây vẫn là dữ liệu tổng hợp cùng các họ triệu chứng, không phải đánh giá ngoài phân phối.

Trước khi chạy lại core đã hoàn thành, sao lưu artefact nếu cần so sánh nhiều phiên.
NB3 huấn luyện lại correct; NB4 bỏ qua adapter đã tồn tại trừ FORCE_RETRAIN=1.
Không trộn baselines hoặc adapter thuộc các model/corpus khác nhau.
`scripts/check_mask_agreement.py` có thể trả exit 1 vì API assistant_masks của
tokenizer rỗng; training trong lab dùng labels từ labkit, không dùng API bị lỗi đó.

Bonus B5 chỉ thực hiện khi người học quyết định repo công khai và đã đăng nhập HF.
Adapter `adapters/correct` sẵn sàng upload; không upload merged model hoặc token.
Lệnh upload, sau khi chọn repo công khai và đăng nhập:

```powershell
.\.venv\Scripts\python.exe scripts/upload_adapter.py account/lab21-qwen35-triage-vi
```

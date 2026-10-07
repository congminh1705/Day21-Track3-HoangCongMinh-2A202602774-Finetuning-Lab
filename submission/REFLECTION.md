# Reflection — Lab 21

Hoàng Công Minh · 2A202602774. Bản phản tư được AI hỗ trợ soạn từ dấu vết thực hiện
thật trong phiên làm bài; không giả định trải nghiệm cá nhân chưa được người học kể.

**1. Điểm bất ngờ của thí nghiệm.** Template giữ được nội dung `<think>`, nhưng API
`return_assistant_tokens_mask` lại trả toàn số 0 vì không có marker generation.
Hai điều tưởng như cùng kiểm tra “template đúng” thực ra chứng minh hai việc khác
nhau. Chỉ giải mã ngược labels của chính dataset dùng để train mới xác nhận được
phần câu trả lời nằm trong loss; mẫu NB1 có 37/94 token được giám sát.

**2. Thời gian tập trung ở đâu.** Phần chờ dài nằm ở tải bộ CUDA và các run GPU,
không phải điền mã vào TODO, vì source đã có pipeline hoàn chỉnh. Một khoản phát
sinh là phải chạy lại huấn luyện sau khi phát hiện seed chưa được đặt trước khởi
tạo LoRA. Run đầu được giữ trong thư mục pilot để không trộn số liệu với run chính.
Điều này cho thấy cần kiểm tra thời điểm áp dụng cấu hình, không chỉ giá trị cấu hình.

**3. Nhận định cần điều chỉnh.** Tên cột `final_loss` không đảm bảo đó là loss của
step cuối: source lấy `Trainer.training_loss`, tức trung bình cả quá trình. Và một
ca fine-tune có điểm thấp không tự động là ca thua baseline; phải chấm cả hai trên
cùng ticket. Vì vậy báo cáo dùng đường loss theo step và delta từng mẫu so với (b),
thay vì chọn những câu trả lời trông đẹp hoặc những con số huấn luyện thuận lợi.

**4. Vai trò và giới hạn của AI assistant.** AI đọc hướng dẫn/rubric, kiểm tra GPU,
cài môi trường, bổ sung lưu dự đoán baseline, audit mask, chạy thí nghiệm và soạn
báo cáo từ JSON/CSV. Ban đầu AI coi `seed=42` trong SFTConfig là đủ; đọc source TRL
cho thấy LoRA được tạo trước khi Trainer đặt seed, nên phải sửa và chạy lại. Test
suite có thể xanh mà vẫn bỏ sót thứ tự khởi tạo này; kiểm tra source thư viện và
bằng chứng thực thi vẫn cần thiết. Dữ liệu bonus phòng lab do AI tạo là dữ liệu
tổng hợp, không thể gọi là ticket thật hoặc khẳng định nó mới hoàn toàn với base.

**5. Bước đầu với khách hàng thật.** Xác định schema, chi phí từng loại lỗi và phạm
vi ứng dụng trước, rồi xây tập đánh giá từ ticket thực tế đã loại thông tin cá
nhân và tách khỏi train. Đo baseline với prompt tốt trên tập đó trước khi xin GPU
huấn luyện. Bổ sung các ca mơ hồ, sản phẩm chưa thấy và kiểm thử ngoài miền; sau
đó mới quyết định fine-tuning có đem lại lợi ích đủ lớn so với prompt hay không.

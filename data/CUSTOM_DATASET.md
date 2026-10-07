# Bonus B2 — Hỗ trợ kỹ thuật phòng lab máy tính

Miền được người học chọn trong phiên làm bài: hỗ trợ kỹ thuật phòng lab máy tính.
Bộ dữ liệu được tạo bằng `scripts/make_lab_support_data.py`; đây là dữ liệu tổng hợp,
không phải ticket của khách hàng thật, không chứa tên người hoặc thông tin đăng nhập.

Có 375 mẫu trong corpus huấn luyện, 75 mẫu đánh giá riêng. Mỗi mẫu có JSON gồm
intent, urgency, product, sentiment. Năm intent: mất kết nối, lỗi phần mềm, thiết bị
lỗi, tài khoản và yêu cầu cài đặt. Năm loại máy và ba thái độ được kết hợp với năm
triệu chứng mỗi intent. Mức khẩn cấp dựa trên bối cảnh: đang thi, đang thực hành,
chuẩn bị buổi sau. Nhãn được tạo từ cấu trúc của tình huống, không dùng model làm judge.

Khử nhiễm: câu đánh giá dùng cách diễn đạt riêng và mã thiết bị EVAL riêng. Generator
kiểm tra uniqueness và giao train/eval rỗng theo toàn bộ input; checksum SHA256 lưu
trong `data/lab_support/checksums.json`. Corpus train được split seed 42 theo tỉ lệ
90/10. Baseline và checksum được đóng băng trước khi train adapter `lab_support`.
Các file eval và train của bài chính không bị thay thế.

Phân phối đích là yêu cầu kỹ thuật tiếng Việt với nhãn nghiệp vụ và bối cảnh lớp học,
khác ticket đổi trả/giao hàng của corpus chính. Không có bằng chứng rằng base chưa
từng thấy các khái niệm này khi pretrain. Vì train và eval vẫn dùng chung các họ
triệu chứng và sản phẩm, điểm đo chỉ phản ánh khả năng chuyển sang cách diễn đạt mới
trong miền hẹp, chưa chứng minh tổng quát hóa trên ticket thực tế. Dữ liệu templated
không thay thế thẩm định thủ công bởi nhân viên phòng lab; cần bổ sung ticket thật,
ca mơ hồ và kiểm thử trên phòng/môn học mới trước khi triển khai.

"""Deterministic synthetic lab-support corpus, isolated from the core lab."""
import hashlib
import itertools
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRODUCTS = ["máy trạm Windows", "máy trạm Linux", "máy tính giảng viên", "laptop thực hành", "máy trạm đồ họa"]
SYMPTOMS = {
    "mat_ket_noi": ["không kết nối được Wi-Fi", "cổng LAN báo mất mạng", "không truy cập được mạng nội bộ",
                    "mất kết nối Internet liên tục", "không nhận địa chỉ IP từ mạng"],
    "loi_phan_mem": ["Python báo lỗi khi khởi động", "VS Code bị treo", "Jupyter không mở được notebook",
                     "trình biên dịch C++ tự thoát", "ứng dụng mô phỏng báo lỗi khi mở"],
    "thiet_bi_loi": ["bàn phím không nhận phím", "màn hình nhấp nháy", "chuột không di chuyển",
                    "máy không bật nguồn", "quạt kêu lớn và máy tự tắt"],
    "tai_khoan": ["tài khoản sinh viên bị khóa", "quên mật khẩu đăng nhập", "tài khoản không có quyền mở thư mục lab",
                  "đăng nhập báo tài khoản hết hạn", "không xác thực được tài khoản sinh viên"],
    "yeu_cau_cai_dat": ["cần cài Python cho môn học", "xin cài VS Code", "cần bổ sung trình biên dịch C++",
                       "xin cài Jupyter", "cần cài ứng dụng mô phỏng mới"],
}
CONTEXTS = [("Buổi thi đang diễn ra và tôi chưa thể tiếp tục", "cao"),
            ("Tôi đang thực hành trong giờ học", "trung_binh"),
            ("Tôi chuẩn bị cho buổi học tuần sau", "thap")]
TONES = [("Tôi rất bực mình vì chuyện này", "tieu_cuc"),
         ("Tôi gửi thông tin để bộ phận kỹ thuật kiểm tra", "trung_tinh"),
         ("Cảm ơn đội kỹ thuật, tôi hài lòng với hỗ trợ trước đây", "tich_cuc")]


def main():
    folder = ROOT / "data" / "lab_support"
    folder.mkdir(exist_ok=True)
    rows = []
    for index, (intent, product, variant, tone) in enumerate(itertools.product(SYMPTOMS, PRODUCTS, range(5), TONES)):
        context, urgency = CONTEXTS[variant % 3]
        label = dict(intent=intent, urgency=urgency, product=product, sentiment=tone[1])
        ticket = f"Tại phòng lab, {product} mã LAB{index:04d} {SYMPTOMS[intent][variant]}. {context}. {tone[0]}."
        rows.append(dict(instruction="Phân loại yêu cầu hỗ trợ phòng lab thành JSON.", input=ticket,
                         output=json.dumps(label, ensure_ascii=False)))
    # All eval sentences use a different wording; no original input is reused.
    evaluation = []
    for index, (intent, product, tone) in enumerate(itertools.product(SYMPTOMS, PRODUCTS, TONES)):
        context, urgency = CONTEXTS[index % 3]
        ticket = (f"Nhờ hỗ trợ {product} (thiết bị EVAL{index:04d}): tình trạng hiện tại là "
                  f"{SYMPTOMS[intent][(index + 2) % 5]}. Bối cảnh: {context}. Phản hồi: {tone[0]}.")
        evaluation.append(dict(input=ticket, label=dict(intent=intent, urgency=urgency,
                                                       product=product, sentiment=tone[1])))
    train_inputs = {r["input"] for r in rows}
    eval_inputs = {r["input"] for r in evaluation}
    assert len(train_inputs) == len(rows) and train_inputs.isdisjoint(eval_inputs)
    for name, records in (("train_seed.jsonl", rows), ("eval_target.jsonl", evaluation)):
        (folder / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                                  encoding="utf-8", newline="\n")
    sys.path.insert(0, str(ROOT / "src"))
    from labkit.data import split
    train_rows, val_rows = split(rows, seed=42)
    split_folder = folder / "split"
    split_folder.mkdir(exist_ok=True)
    for name, records in (("train.jsonl", train_rows), ("val.jsonl", val_rows)):
        (split_folder / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
            encoding="utf-8", newline="\n")
    checks = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
              for name in ("train_seed.jsonl", "eval_target.jsonl")}
    (folder / "checksums.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(f"custom lab support: {len(rows)} train pool, {len(evaluation)} eval", flush=True)


if __name__ == "__main__":
    main()

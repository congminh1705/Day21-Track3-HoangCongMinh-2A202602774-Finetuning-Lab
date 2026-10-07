"""Write a measured report; never substitute missing experimental results."""
from __future__ import annotations

import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labkit import report
from labkit.config import get_tier


def read(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def main():
    proof, stats, template = (read(n) for n in ("mask_proof.json", "token_stats.json", "template_check.json"))
    frozen, verdict, autopsy, qualitative = (read(n) for n in
        ("baselines_frozen.json", "verdict.json", "autopsy.json", "qualitative.json"))
    runs = {r["run"]: r for r in report.read_rows()}
    joined = [dict(runs[row["run"]], target=row["target"]) for row in autopsy]
    target_qualitative = list(qualitative)
    if (ROOT / "results" / "qualitative_regression.json").exists():
        qualitative = qualitative + read("qualitative_regression.json")
    worst = [r for r in qualitative if r["outcome"] == "loss"][:2]
    best = [r for r in reversed(qualitative) if r["outcome"] == "win"][:2]
    selected = worst + best
    for row in sorted(target_qualitative, key=lambda r: r["ft_score"]) + qualitative:
        if row not in selected and len(selected) < 5:
            selected.append(row)
    selected = selected[:5]
    examples = []
    for r in selected:
        def cell(s):
            return str(s).replace("|", "\\|").replace("\n", " ")
        examples.append({"i": r["i"], "nhóm": r.get("group", "target"), "ticket": cell(r["ticket"]),
            "nhãn": cell(json.dumps(r["label"], ensure_ascii=False)),
            "baseline_b": cell(r["baseline_pred"]), "fine_tune": cell(r["ft_pred"]),
            "b_score": r["baseline_score"], "ft_score": r["ft_score"], "kết_quả": r["outcome"]})
    gate = verdict["verdict"]
    correct, attention, low_lr, quant = (runs[k] for k in ("correct", "attn_only", "wrong_lr", "qlora"))
    placement = {r["run"]: r["target"] for r in autopsy}
    target_order = sorted(placement, key=lambda k: placement[k], reverse=True)
    loss_order = sorted(placement, key=lambda k: float(runs[k]["final_loss"]))
    curves = {k: [row["loss"] for row in read(f"{k}_training_log.json") if "loss" in row]
              for k in ("correct", "wrong_lr")}
    qualitative_losses = [r for r in target_qualitative if r["ft_score"] < 1.0]
    error_fields = {}
    from labkit.evaluate import _parse_json_loose, normalize
    for example in qualitative_losses:
        parsed = _parse_json_loose(example["ft_pred"]) or {}
        for key, expected in example["label"].items():
            if normalize(str(parsed.get(key, ""))) != normalize(str(expected)):
                error_fields[key] = error_fields.get(key, 0) + 1
    memory_saved = float(correct["peak_vram_gb"]) - float(quant["peak_vram_gb"])
    tier = get_tier()
    environment = read("environment.json")
    text = f"""# Lab 21 — Bằng chứng và quyết định triển khai

Hoàng Công Minh · MSSV 2A202602774 · Phiên chạy 07–08/10/2026 (Asia/Bangkok).
Thực hiện với AI assistant; các con số dưới đây lấy từ artefact chạy thật.

## Thiết kế và môi trường

Model: `{frozen['model']}`; tier `{frozen['tier']}`. Máy thực tế là
{environment['name']} {environment['vram_gb']} GB, Windows, Python {environment['python']}.
Precision bf16 phần cứng: {environment['bf16']}. Model 0.8B được chọn vì model 4B mặc định
không vừa GPU này; đây là thay đổi được README cho phép qua BASE_MODEL.
Revision model đã chạy: `{environment['base_revision']}`.
Giữ cấu hình tier LAPTOP: batch mỗi thiết bị 1, gradient accumulation 8,
batch hiệu dụng 8, seed 42. Corpus chính là 250 ticket CSKH tiếng Việt → JSON
intent/urgency/product/sentiment, split 225 train và 25 validation. Chọn corpus này
để phép chấm khách quan và so sánh lại được với hướng dẫn, không cần LLM judge.
Validation được tách riêng nhưng không dùng để lựa chọn checkpoint; tất cả run
được chấm ở cuối cùng ngân sách, không tìm checkpoint tốt nhất trên eval.

Độ dài đo được: p95={stats['p95']}, p99={stats['p99']}, max={stats['max']};
khuyến nghị làm tròn là {stats['suggested_max_length']}. `max_length={tier.max_length}`
giữ nguyên tier như hướng dẫn đổi base. Dataset pre-tokenized không pad đến giới hạn
này, batch=1 nên không trả chi phí padding tới {tier.max_length}; không mẫu nào bị cắt.
Epochs=2; ngân sách của mỗi run là {correct['max_steps']} optimizer steps.
Phiên bản cài thực tế được lưu trong `submission/requirements-lock.txt`.

## Chứng minh mask và template

`answer_is_supervised={proof['answer_is_supervised']}`;
`question_is_masked={proof['question_is_masked']}`.
Mask `{proof['mask_mode']}` tính loss trên {proof['n_supervised']}/{proof['n_total']}
token của mẫu kiểm chứng, supervised_fraction={proof['supervised_fraction']}.

```text
{proof['supervised_preview']}
```

Template giữ reasoning: `{template.get('ok')}`; body hiện diện:
`{template.get('body_present')}`. Nhưng template không có marker generation để API
assistant mask tự suy ra loss span. Kiểm tra tokenizer trả mask rỗng; đây là cảnh báo
về đường API đó, không phải mask đã chứng minh của labkit bị sai. Huấn luyện dùng
input_ids và labels từ build_example, có EOS được giám sát, packing tắt. Không dùng
assistant_only_loss để tránh thay template hoặc học trên mask khác với NB1.

## Mốc đóng băng trước huấn luyện

NB2 chạy trước NB3. Không thay OPTIMIZED_PROMPT hay hai tập eval sau khi thấy kết quả
huấn luyện. SHA prompt (b): `{frozen['optimized_prompt_sha']}`. Chấm đầy đủ
{frozen['n_target']} target và {frozen['n_regression']} regression; smoke_mode:
`{frozen['smoke_mode']}`. Dự đoán gốc được lưu trong baseline_predictions.json.
Điểm (b) {'cao hơn' if frozen['baseline_b']['target'] > frozen['baseline_a']['target'] else 'không cao hơn'}
(a); không làm yếu baseline để nâng thành tích fine-tune.
Checksum dữ liệu gốc khớp Git HEAD và checksums.json khi dùng LF chuẩn. Working tree
Windows dùng CRLF nên hash byte thô ban đầu cảnh báo khác; verifier chỉ chuẩn hóa
ký tự xuống dòng, vẫn phát hiện thay nhãn hoặc nội dung. Bằng chứng từng file nằm
trong data_integrity_audit.json. Không đổi nội dung train/eval của thí nghiệm chính.

{report.markdown_table(verdict['comparison'])}

Format ở đây là tỷ lệ khóa cần thiết mà parser của lab tìm được, cho phép trích JSON
trong prose/fence; không phải tỷ lệ chỉ xuất một JSON sạch. Target là trung bình tỷ lệ
đúng của bốn trường. Regression là keyword recall trên 15 câu hỏi; nó chỉ là phép thử
hồi quy nhỏ, không bao phủ toàn bộ năng lực tổng quát. Latency là trung bình mỗi mẫu
với greedy decode, batch inference 4, cùng máy; chưa phải p95 khi phục vụ một request.

## Bốn cấu hình, một ngân sách step

{report.markdown_table(joined, ['run','target','r','trainable_params','learning_rate','final_loss','max_steps','train_seconds','peak_vram_gb'])}

### Vị trí so với rank

Run attn_only chỉ đổi vị trí gắn vào q/v; rank được nâng tới {attention['r']} để khớp
ngân sách {attention['trainable_params']} so với {correct['trainable_params']} của correct.
Độ lệch ngân sách là {abs(int(attention['trainable_params']) / int(correct['trainable_params']) - 1):.4%}.
Target attn_only={placement['attn_only']}, correct={placement['correct']}; chênh lệch
{placement['attn_only']-placement['correct']:+.4f}. Đây là phép đo ảnh hưởng vị trí trong
điều kiện đã khống chế tổng tham số, không phải bằng chứng rằng rank cao luôn tốt.
Loss tương ứng là {attention['final_loss']} và {correct['final_loss']}.
Thứ tự target giảm dần: {' → '.join(target_order)} (đọc giá trị trong bảng để nhận biết hòa).
Thứ tự loss tăng dần: {' → '.join(loss_order)}.
Hai thứ tự {'trùng nhau' if target_order == loss_order else 'khác nhau'};
chỉ số huấn luyện không đủ để kết luận tổng quát hóa.
Không suy rộng kết luận từ một model, một seed và corpus nhỏ sang mọi kiến trúc.
Kiến trúc gồm 18 lớp linear attention và 6 lớp full attention (architecture.json).
q_proj/v_proj thuộc các lớp full attention; text-linear còn gắn vào projection của
linear attention và MLP. Tăng rank q/v không thay được phạm vi lớp được tác động.
Lưu ý: cột final_loss trong source là `Trainer.training_loss` trung bình cả run,
không phải loss duy nhất của step cuối. Đường loss theo step mới dùng để chẩn đoán.

### Thang learning rate

Wrong_lr giữ r=16, vị trí text-linear, cùng dữ liệu và step; chỉ giảm LR từ
{correct['learning_rate']} xuống {low_lr['learning_rate']}. Loss được lưu theo step
trong correct_training_log.json và wrong_lr_training_log.json để phân biệt giảm chậm,
dao động và thực sự không học. Loss log đầu/cuối của correct là
{curves['correct'][0]:.4f}/{curves['correct'][-1]:.4f}; wrong_lr là
{curves['wrong_lr'][0]:.4f}/{curves['wrong_lr'][-1]:.4f}. Loss trung bình là {correct['final_loss']} so với
{low_lr['final_loss']}, target là {placement['correct']} so với {placement['wrong_lr']}.
Không thể quy mọi chênh lệch loss cho năng lực adapter: ở đây thang cập nhật khác nhau.
Phải nhìn target và regression cùng với đường loss trước khi chọn một cấu hình.

### Chi phí lượng tử hóa

QLoRA chỉ đổi base sang NF4 4-bit; cùng vị trí, r, LR, dữ liệu và step.
Peak VRAM: {correct['peak_vram_gb']} GB so với {quant['peak_vram_gb']} GB,
tiết kiệm tuyệt đối {memory_saved:.4f} GB. Target thay đổi
{placement['qlora']-placement['correct']:+.4f}; thời gian chạy và loss có trong bảng.
Adapter QLoRA được đánh giá trên base 4-bit đúng với khi train. Số đo này chỉ kiểm tra
khuyến nghị về lượng tử hóa trên model 0.8B của phiên chạy, không chứng minh kết quả
trên Qwen3.5-4B/T4. VRAM là max_memory_allocated của PyTorch, khác tổng bộ nhớ nvidia-smi.

## Phán quyết và ý nghĩa

**{'PASSED' if gate['passed'] else 'FAILED'}**. Target Δ={gate['target_delta']:+.4f};
regression Δ={gate['regression_delta']:+.4f}; valid_trace_rate của phép đo core
(thinking tắt)={verdict['valid_trace_rate']}. Các lý do do gate ghi:

{chr(10).join('- ' + reason for reason in gate['reasons'])}

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

{report.markdown_table(examples)}

Trên target có {sum(r['outcome']=='loss' for r in target_qualitative)} ca fine-tune thua baseline.
Tổng trên cả target và regression: thua {sum(r['outcome']=='loss' for r in qualitative)};
thắng: {sum(r['outcome']=='win' for r in qualitative)};
hòa: {sum(r['outcome']=='tie' for r in qualitative)}.
{'Đã đưa ít nhất hai ca thua vào bảng.' if len(worst)>=2 else 'Không có đủ hai ca thua quan sát được; không tạo ví dụ giả để đạt rubric.'}
Toàn bộ dự đoán, nhãn và score của từng mẫu được lưu không cắt ngắn trong qualitative.json.
Các ca thua ngoài miền nằm trong qualitative_regression.json và được đánh dấu nhóm
regression trong bảng; đó là câu hỏi kiến thức/chỉ dẫn, không phải ticket target.
Ví dụ của nhóm này dùng keyword recall, khác field accuracy của nhóm target.
Keyword recall có giới hạn: ở câu đổi 1 km, baseline chứa 1000 nên được điểm, nhưng
còn kèm diễn giải sai “m = 100”. Không coi điểm keyword cao là bằng chứng mọi câu
văn đều đúng; vẫn giữ nguyên output để người đọc kiểm tra giới hạn của phép chấm.
Số trường sai trên target (kể cả khi vẫn thắng baseline): {json.dumps(error_fields, ensure_ascii=False)}.
Các số này mô tả lỗi quan sát được, không chứng minh nguyên nhân bằng quan sát đơn lẻ.

## Kết luận và điều rút ra

Quyết định triển khai phải dựa vào baseline tốt nhất đã đo, vì người dùng có thể đạt
hành vi mong muốn bằng một prompt mà không cần bảo trì trọng số mới. Trong phiên
này, gate {'đạt' if gate['passed'] else 'không đạt'} và đó là kết luận về phép đo hiện có,
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

"""
    bonus_csv = ROOT / "results" / "bonus_runs.csv"
    if bonus_csv.exists():
        with bonus_csv.open(encoding="utf-8", newline="") as handle:
            bonus_runs = list(csv.DictReader(handle))
        text += report.markdown_table(bonus_runs, ["run", "r", "lora_alpha", "trainable_params",
            "learning_rate", "max_steps", "mask_mode", "optimizer", "train_seconds",
            "peak_vram_gb"]) + "\n\n"
    for filename, title in (("merge_check.json", "B1 — Merge"), ("hot_swap.json", "B1 — Hot-swap"),
        ("custom_baselines_frozen.json", "B2 — Baseline miền phòng lab"),
        ("custom_verdict.json", "B2 — Kết quả miền phòng lab"),
        ("trace_mask_proof.json", "B3 — Mask có khác nhau thật không"),
        ("trace_input_audit.json", "B3 — Token đầu vào giống nhau"),
        ("reasoning_contrast.json", "B3 — Thinking bật"), ("rank_sweep.json", "B4 — Rank cố định vị trí"),
        ("muon_preregistered.json", "B6 — Dự đoán trước khi chạy"),
        ("muon_result.json", "B6 — Optimizer Muon"),
        ("muon_output_diagnostic.json", "B6 — Chẩn đoán output (bổ sung)")):
        path = ROOT / "results" / filename
        if path.exists():
            text += f"### {title}\n\n```json\n{json.dumps(read(filename), ensure_ascii=False, indent=2)}\n```\n\n"
        else:
            text += f"{title}: chưa có kết quả chạy, không nhận là đã hoàn thành.\n\n"
    if (ROOT / "results" / "custom_verdict.json").exists():
        custom = read("custom_verdict.json")
        custom_gate = custom["verdict"]
        text += (f"B2 huấn luyện {custom['n_train']} mẫu, giữ lại {custom['n_val']} validation; "
                 "75 mẫu eval có cách diễn đạt và mã máy riêng. "
                 f"Target so với baseline tối ưu đổi {custom_gate['target_delta']:+.4f}; "
                 f"regression đổi {custom_gate['regression_delta']:+.4f}; "
                 f"gate {'PASSED' if custom_gate['passed'] else 'FAILED'}. "
                 "Không đổi ngưỡng gate cho miền riêng. Corpus này mới về cách kết hợp nhãn "
                 "và bối cảnh phòng học so với corpus CSKH của lab, nhưng không chứng minh "
                 "base chưa gặp khái niệm kỹ thuật khi pretrain. Kết quả chỉ kiểm tra chuyển "
                 "sang schema nghiệp vụ trong miền tổng hợp hẹp.\n\n")
    if (ROOT / "results" / "rank_sweep.json").exists():
        sweep = read("rank_sweep.json")
        amplitudes = {"rank": max(r["target"] for r in sweep) - min(r["target"] for r in sweep),
            "vị trí": abs(placement["attn_only"] - placement["correct"]),
            "LR": abs(placement["wrong_lr"] - placement["correct"])}
        text += "Biên độ ảnh hưởng trên target: " + json.dumps(amplitudes, ensure_ascii=False) + ".\n\n"
        groups = []
        for knob in sorted(amplitudes, key=amplitudes.get, reverse=True):
            if groups and round(amplitudes[groups[-1][0]], 4) == round(amplitudes[knob], 4):
                groups[-1].append(knob)
            else:
                groups.append([knob])
        text += "Thứ tự theo biên độ: " + " > ".join(" = ".join(group) for group in groups) + ".\n\n"
        by_rank = {r["rank"]: r for r in sweep}
        if 64 in by_rank and 16 in by_rank:
            text += (f"Tăng r=16 lên r=64 dùng gấp 4 ngân sách tham số LoRA, target tăng "
                     f"{by_rank[64]['target']-by_rank[16]['target']:+.4f}, regression đổi "
                     f"{by_rank[64]['regression']-by_rank[16]['regression']:+.4f}. "
                     "r=16 đã gần trần target của tập này; lợi ích rank lớn phải được cân với "
                     "bộ nhớ, khả năng ngoài miền và đánh giá trên dữ liệu mới.\n\n")
        text += ("Corpus nhỏ có thể chưa dùng hết năng lực r=64; phải nhìn điểm giữ lại thay vì "
                 "giả định nhiều tham số thì tốt hơn. Không có learning curve theo kích thước dữ liệu "
                 "nên chưa thể xác định ngưỡng thông tin mà rank lớn trở nên cần thiết.\n\n")
    if (ROOT / "results" / "reasoning_contrast.json").exists():
        reasoning = read("reasoning_contrast.json")
        text += report.markdown_table([
            {"MASK_MODE": row.get("mask_mode", "base (chưa fine-tune)"),
             "target": row["target"], "valid_trace_rate": row["valid_trace_rate"],
             "regression": row["regression"]} for row in reasoning]) + "\n\n"
        if reasoning[0]["valid_trace_rate"] == 0:
            text += ("Base thinking có valid_trace_rate=0 trong ngân sách 512 token; "
                     "không có mốc trace hợp lệ cao trước train để chứng minh suy giảm so với base. "
                     "Output gốc được giữ trong base_bonus_predictions.json để kiểm tra "
                     "lặp/cắt ngắn. So sánh hai mask vẫn đo hành vi trace, nhưng không đủ "
                     "để tuyên bố tái lập mất năng lực suy luận từ base.\n\n")
        a, b = reasoning[1:3]
        text += (f"B3 response-only so với assistant-only: target Δ={b['target']-a['target']:+.4f}, "
                 f"trace Δ={b['valid_trace_rate']-a['valid_trace_rate']:+.4f}. "
                 "Nhìn target một mình không đo được hình thức reasoning. "
                 + ("Trong phiên này có target tăng đồng thời trace giảm. " if b['target'] > a['target']
                    and b['valid_trace_rate'] < a['valid_trace_rate'] else
                    "Phiên này không quan sát thấy cặp dấu target tăng/trace giảm giữa hai mask. ")
                 + "Đây là kết quả cần báo cáo nguyên trạng.\n\n")
        text += ("Ở response-only, trace vẫn nằm trong context khi train (teacher forcing), "
                 "nhưng token của trace không được giám sát. Khi eval thinking bật, model "
                 "phải tự sinh phần context đó trước JSON; loss JSON thấp trên context có sẵn "
                 "không bảo đảm sinh được toàn chuỗi. Đây là một cơ chế khả dĩ cho chênh lệch "
                 "đã đo, chưa được tách riêng bằng thí nghiệm can thiệp. Không so trực tiếp "
                 "loss giữa hai mask như cùng một mục tiêu vì số token được giám sát khác nhau.\n\n")
    if (ROOT / "results" / "muon_result.json").exists():
        muon = read("muon_result.json")
        text += (f"B6 target Muon Δ={muon['target']-placement['correct']:+.4f} so với AdamW. "
                 "Giả thuyết Muon có thể kém hơn được viết trước khi train; không thay đổi dự đoán sau "
                 "khi thấy điểm. Vì pretraining optimizer chưa được xác minh và LR cũng đổi theo họ "
                 "optimizer, thí nghiệm này không tách riêng tác động optimizer mismatch.\n\n")
        text += ("Dự đoán target thấp hơn AdamW "
                 + ("phù hợp với kết quả quan sát. " if muon['target'] < placement['correct'] else
                    "không được kết quả quan sát hỗ trợ. ")
                 + "Ngân sách step nhỏ, các ma trận adapter và lựa chọn LR đều có thể ảnh hưởng; "
                 "không có ablation đủ để gán nguyên nhân cho một yếu tố.\n\n")
        if (ROOT / "results" / "muon_output_diagnostic.json").exists():
            diagnostic = read("muon_output_diagnostic.json")
            text += (f"B6 có {diagnostic['n_first_object_parseable']}/{diagnostic['n']} output "
                     f"đọc được object đầu và {diagnostic['n_extra_text_after_first_object']} output "
                     "còn nội dung phía sau; output lặp JSON tới giới hạn token. "
                     f"Chỉ chấm object đầu cho field accuracy={diagnostic['first_object_field_accuracy']:.4f}, "
                     "nhưng đây là chẩn đoán bổ sung, không thay phép chấm chính. Target/format chính "
                     "vẫn bằng 0 vì parser không nhận chuỗi nhiều JSON. Do đó số 0 không chứng minh "
                     "model không biết phân loại; lỗi hành vi kết thúc output làm hỏng giao thức trả lời. "
                     "Giữ output gốc và scorer chính để phép so sánh không bị chỉnh sau kết quả.\n\n")
    text += """Merge bỏ overhead adapter trong đồ thị suy luận nhưng làm mất khả năng chọn
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
"""
    (ROOT / "submission" / "REPORT.md").write_text(text, encoding="utf-8")
    model_card = f"""---
base_model: {frozen['model']}
library_name: peft
pipeline_tag: text-generation
language:
- vi
tags:
- lora
- lab21
---
# Lab 21 Vietnamese customer-support triage adapter

Student: Hoàng Công Minh, 2A202602774. Base: {frozen['model']}.
Base revision: {environment['base_revision']}.
LoRA rank 16, alpha 32, text-decoder linear layers; learning rate 1e-4,
58 optimizer steps, seed 42, assistant-only labels verified by token decoding.
Training: 225 synthetic seed tickets; 25 held out for validation.
Evaluation: 50 task examples and 15 general-capability prompts, greedy decoding.
Frozen optimized-prompt baseline target: {frozen['baseline_b']['target']}.
Fine-tuned target: {placement['correct']}; gate: {'PASSED' if gate['passed'] else 'FAILED'}.

Use the same base checkpoint, tokenizer and short system instruction
`Phân loại ticket sau.`. Outputs use intent, urgency, product, sentiment.
This is an educational experiment, evaluated on a small synthetic corpus.
The gate measures relative task accuracy and a limited regression probe;
it does not establish readiness for production support workflows.
Full evidence is in submission/REPORT.md and results/ in the accompanying lab archive.
"""
    (ROOT / "adapters" / "correct" / "README.md").write_text(model_card, encoding="utf-8")


if __name__ == "__main__":
    main()

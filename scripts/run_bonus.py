"""Controlled B2/B3/B4/B6 experiments, preserving every core artifact.

Usage: python scripts/run_bonus.py rank|trace|custom|muon|all
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labkit import data, evaluate as ev, generate, modeling, report, train
from labkit.config import SPECS, get_tier, training_epochs


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fit(key, records, *, rank=16, mode="assistant-only", thinking=None, optimizer="adamw", lr=None):
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer

    tier = get_tier()
    spec = dataclasses.replace(SPECS["correct"], key=key, r=rank, alpha=2 * rank,
                               lr=lr if lr is not None else SPECS["correct"].lr,
                               label=f"text-linear · r={rank} · optimizer={optimizer} · {mode}")
    out = ROOT / "adapters" / key
    if (out / "adapter_model.safetensors").exists():
        print(f"reuse {key}", flush=True)
        return out
    set_seed(42)
    model, tok = generate.load_base(tier)
    rows = data.to_training_dataset(tok, records, max_length=tier.max_length,
                                    mask_mode=mode, enable_thinking=thinking)
    targets = modeling.resolve_target_modules(model, spec.target)
    trainable = modeling.count_lora_params(model, targets, rank)
    steps = train.planned_steps(len(rows), tier, training_epochs())
    kw, _ = train.filter_kwargs(SFTConfig, train.sft_config_kwargs(
        tier, spec, str(out), max_steps=steps))
    optimizer_kw = {}
    if optimizer == "muon":
        import torch
        optimizer_kw["optimizer_cls_and_kwargs"] = (torch.optim.Muon,
            {"lr": spec.lr, "weight_decay": 0.01, "momentum": 0.95,
             "adjust_lr_fn": "match_rms_adamw"})
    set_seed(42)
    trainer = SFTTrainer(model=model, args=SFTConfig(**kw),
                         train_dataset=Dataset.from_list(rows), processing_class=tok,
                         peft_config=LoraConfig(**train.lora_config_kwargs(spec, targets)),
                         **optimizer_kw)
    if optimizer == "muon":
        assert all(p.ndim == 2 for p in trainer.model.parameters() if p.requires_grad)
    train.align_trainable_precision(trainer.model)
    generate.free_memory()
    start = time.perf_counter()
    res = trainer.train()
    elapsed = time.perf_counter() - start
    trainer.model.save_pretrained(out)
    tok.save_pretrained(out)
    row = train.summarize_run(spec, tier, targets,
                              trainable,
                              elapsed, generate.peak_vram_gb())
    row.update(final_loss=res.training_loss, max_steps=steps, mask_mode=mode, optimizer=optimizer)
    report.append_row(row, filename="bonus_runs.csv")
    report.write_json(trainer.state.log_history, f"{key}_training_log.json")
    del trainer, model
    generate.free_memory()
    return out


def score(key, *, thinking=False, target_path=None, system_prompt=None):
    from peft import PeftModel
    from labkit.config import NAIVE_PROMPT
    model, tok = generate.load_base(get_tier())
    if key != "base":
        model = PeftModel.from_pretrained(model, str(ROOT / "adapters" / key))
    model.eval()
    target = read_rows(target_path or ROOT / "data" / "eval_target.jsonl")
    reg = read_rows(ROOT / "data" / "eval_regression.jsonl")
    preds, latency = generate.generate_batch(model, tok, [r["input"] for r in target],
        system=system_prompt or NAIVE_PROMPT, enable_thinking=thinking,
        max_new_tokens=512 if thinking else 160, label=key)
    # The opening tag belongs to the generation prompt on Qwen, outside new tokens.
    traces = [("<think>\n" + p if thinking and "<think>" not in p else p) for p in preds]
    reg_preds, _ = generate.generate_batch(model, tok, [r["instruction"] for r in reg],
        system=None, enable_thinking=False, max_new_tokens=96, label=f"{key}/regression")
    result = {"run": key, "thinking": thinking, "n": len(target),
        "max_new_tokens": 512 if thinking else 160,
        "target": sum(ev.triage_field_accuracy(p, r["label"]) for p, r in zip(preds, target)) / len(target),
        "regression": sum(ev.keyword_recall(p, r["keywords"]) for p, r in zip(reg_preds, reg)) / len(reg),
        "format": sum(ev.has_required_keys(p, ev.TRIAGE_KEYS) for p in preds) / len(preds),
        "latency_ms": latency,
        "valid_trace_rate": sum(ev.valid_reasoning_trace(p) for p in traces) / len(traces)}
    prediction_key = key if target_path is None else f"lab_support_{key}"
    report.write_json({"scores": result, "target_predictions": preds,
        "reconstructed_traces": traces, "regression_predictions": reg_preds}, f"{prediction_key}_bonus_predictions.json")
    if key == "muon":
        first = []
        for pred in preds:
            try:
                obj, end = json.JSONDecoder().raw_decode(pred.lstrip())
                first.append((obj, bool(pred.lstrip()[end:].strip())))
            except (ValueError, TypeError):
                first.append((None, False))
        report.write_json({"diagnostic_only": True, "official_scores_unchanged": True,
            "n": len(first), "n_first_object_parseable": sum(o is not None for o, _ in first),
            "n_extra_text_after_first_object": sum(extra for _, extra in first),
            "first_object_field_accuracy": sum(ev.triage_field_accuracy(
                json.dumps(o, ensure_ascii=False), record["label"]) if o is not None else 0
                for (o, _), record in zip(first, target)) / len(target),
            "interpretation": "Supplementary parser diagnostic only; repeated JSON breaks official scorer. Do not substitute for official scores."},
            "muon_output_diagnostic.json")
    if key == "correct" and not thinking and target_path is None:
        baseline = json.loads((ROOT / "results" / "baseline_predictions.json").read_text(encoding="utf-8"))
        qualitative = []
        for i, (record, tuned, original) in enumerate(zip(reg, reg_preds, baseline["regression_b"])):
            ft_score = ev.keyword_recall(tuned, record["keywords"])
            b_score = ev.keyword_recall(original, record["keywords"])
            qualitative.append({"i": f"reg-{i}", "group": "regression", "ticket": record["instruction"],
                "label": {"keywords": record["keywords"]}, "ft_pred": tuned, "baseline_pred": original,
                "ft_score": ft_score, "baseline_score": b_score, "delta": ft_score - b_score,
                "outcome": "loss" if ft_score < b_score else "win" if ft_score > b_score else "tie"})
        report.write_json(sorted(qualitative, key=lambda r: r["delta"]), "qualitative_regression.json")
    del model
    generate.free_memory()
    return result


def rank_sweep():
    records = read_rows(ROOT / "data" / "split" / "train.jsonl")
    scores = []
    for rank in (16, 8, 64):
        key = "correct" if rank == 16 else f"rank_{rank}"
        if rank != 16:
            fit(key, records, rank=rank)
        scores.append(dict(score(key), rank=rank, target_modules="text-linear"))
        report.write_json(scores, "rank_sweep.json")


def trace_contrast():
    import hashlib
    from transformers import AutoTokenizer
    records = read_rows(ROOT / "data" / "split" / "train.jsonl")
    traced = []
    for record in records:
        answer = json.loads(record["output"])
        rationale = (f"Ticket nhắc đến {answer['product']}. Nội dung yêu cầu thuộc nhóm "
                     f"{answer['intent']}; tín hiệu thời gian và thái độ được gán lần lượt "
                     f"{answer['urgency']} và {answer['sentiment']} theo nhãn của bộ dữ liệu.")
        traced.append(dict(record, output=f"<think>{rationale}</think>\n{record['output']}"))
    # Synthetic label-derived traces test mask sensitivity; they do not establish
    # preservation of mathematical reasoning or the published collapse result.
    report.write_json({"source": "train split only; synthetic label-derived explanations",
        "n": len(traced), "records": traced}, "trace_training_data.json")
    tok = AutoTokenizer.from_pretrained(get_tier().model_id)
    proofs = []
    input_hashes = {}
    for mode in ("assistant-only", "response-only"):
        examples = [data.build_example(tok, data.to_messages(r), max_length=get_tier().max_length,
                    mask_mode=mode, enable_thinking=True) for r in traced]
        assert all(e.n_supervised > 0 for e in examples)
        input_hashes[mode] = hashlib.sha256(json.dumps([e.input_ids for e in examples]).encode()).hexdigest()
        proofs.append({"mask_mode": mode, "n": len(examples),
            "supervised_tokens": sum(e.n_supervised for e in examples),
            "preview": data.decode_supervised(tok, examples[0])})
    assert proofs[0]["supervised_tokens"] > proofs[1]["supervised_tokens"]
    report.write_json(proofs, "trace_mask_proof.json")
    assert len(set(input_hashes.values())) == 1, "mask experiment changed input tokens"
    report.write_json({"input_sha256_by_mask": input_hashes, "identical_inputs": True,
        "token_stats": data.token_stats([e.n_total for e in examples])}, "trace_input_audit.json")
    scores = [score("base", thinking=True)]
    for mode in ("assistant-only", "response-only"):
        key = "trace_" + mode.replace("-", "_")
        fit(key, traced, mode=mode, thinking=True)
        scores.append(dict(score(key, thinking=True), mask_mode=mode))
        report.write_json(scores, "reasoning_contrast.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("rank", "trace", "custom", "muon", "all"))
    arg = parser.parse_args()
    if arg.experiment in ("rank", "all"):
        rank_sweep()
    if arg.experiment in ("trace", "all"):
        trace_contrast()
    if arg.experiment in ("custom", "all"):
        import hashlib
        from make_lab_support_data import main as make_data
        make_data()
        folder = ROOT / "data" / "lab_support"
        target_path = folder / "eval_target.jsonl"
        prompt = ('Bạn phân loại yêu cầu hỗ trợ phòng lab máy tính. Chỉ trả về JSON với 4 khóa '
            'intent, urgency, product, sentiment. intent thuộc mat_ket_noi, loi_phan_mem, '
            'thiet_bi_loi, tai_khoan, yeu_cau_cai_dat. urgency: cao nếu thi/kiểm tra đang '
            'diễn ra; trung_binh nếu đang thực hành; thap nếu chuẩn bị buổi sau. sentiment: '
            'tieu_cuc nếu bực mình; trung_tinh nếu thông báo; tich_cuc nếu cảm ơn và hài lòng. '
            'product là tên máy nguyên văn, không bao gồm mã thiết bị.')
        baseline_a = score("base", target_path=target_path)
        baseline_b = score("base", target_path=target_path, system_prompt=prompt)
        report.write_json({"baseline_a": baseline_a, "baseline_b": baseline_b,
            "optimized_prompt": prompt, "eval_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
            "model": get_tier().model_id}, "custom_baselines_frozen.json")
        records = read_rows(folder / "train_seed.jsonl")
        train_records, val = data.split(records, seed=42)
        fit("lab_support", train_records)
        tuned = score("lab_support", target_path=target_path)
        verdict = ev.regression_gate(ev.GroupScores(**{k: tuned[k] for k in
            ("target", "regression", "format", "latency_ms", "n")}),
            ev.GroupScores(**{k: baseline_b[k] for k in
            ("target", "regression", "format", "latency_ms", "n")}))
        report.write_json({"scores": tuned, "verdict": verdict.as_dict(),
            "n_train": len(train_records), "n_val": len(val)}, "custom_verdict.json")
    if arg.experiment in ("muon", "all"):
        prediction_path = ROOT / "results" / "muon_preregistered.json"
        if not prediction_path.exists():
            report.write_json({"hypothesis": "Muon may underperform AdamW on target at this small step budget; LoRA may limit damage.",
                "lr": 0.001, "momentum": 0.95, "adjust_lr_fn": "match_rms_adamw",
                "reason": "Use Muon's own default LR and matrix RMS scaling; do not copy AdamW's LR.",
                "limitation": "Base pretraining optimizer is unverified; this does not isolate optimizer mismatch."},
                "muon_preregistered.json")
        fit("muon", read_rows(ROOT / "data" / "split" / "train.jsonl"), optimizer="muon", lr=0.001)
        report.write_json(score("muon"), "muon_result.json")

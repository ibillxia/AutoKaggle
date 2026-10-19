"""本地小模型 surrogate LoRA 训练器（配置驱动）。

用法
----
    python surrogate_train.py smoke                       # 零下载，验证整条流水线
    python surrogate_train.py features --config configs/local_surrogate.yaml
    python surrogate_train.py train --config configs/local_surrogate.yaml
    python surrogate_train.py train --config configs/local_surrogate.yaml --max-steps 20
    python surrogate_train.py export --adapter-dir runs/local_surrogate --name main_lora

设计原则
--------
- **本地调通、云端只换配置**：model.id / quantization / batch 全部来自 YAML，
  脚本本身在「RTX 4060 Ti 8GB + 1.5B 替身」和「Kaggle L4x4 + 31B 真机」之间零改动。
- **零下载可验证**：``smoke`` 用随机初始化的微型模型 + 字符级分词器，不联网也能
  证明「模板渲染 -> assistant 段 mask -> LoRA 挂载 -> 训练 -> 导出 -> 提交包校验」全通。
- **不假装能干的事**：``checkpoint_native``（直接加载官方 w4a16 权重）这条路径
  本地没有 31B 权重，**未经实机验证**，代码里会明确警告。
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from adapter_export import export as export_adapter, inspect_adapter, scan_extensions  # noqa: E402
from sft_data import (  # noqa: E402
    SftCollator,
    StubTokenizer,
    build_features,
    load_tokenizer,
    load_trajectories,
    make_synthetic_samples,
)

# Gemma / Qwen / Llama 这几族的线性层命名一致，真机换成 Gemma 4 时仍然命中
TARGETS_BY_ARCH = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def _dtype_kwarg() -> str:
    """transformers 5.x 把 torch_dtype 改名成了 dtype。"""
    import transformers

    major = int(transformers.__version__.split(".")[0])
    return "dtype" if major >= 5 else "torch_dtype"


def _torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def _bnb_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:
        return False


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg.setdefault("model", {})
    cfg.setdefault("lora", {})
    cfg.setdefault("data", {})
    cfg.setdefault("training", {})
    cfg["model"].setdefault("quantization", "none")
    cfg["model"].setdefault("compute_dtype", "bfloat16")
    cfg["model"].setdefault("attn_implementation", "sdpa")
    cfg["data"].setdefault("train_on", "assistant_only")
    cfg["data"].setdefault("max_seq_len", 2048)
    cfg["lora"].setdefault("target_modules", "auto")
    return cfg


def resolve_targets(model, spec: Any):
    if isinstance(spec, list):
        return spec
    if isinstance(spec, str) and spec != "auto":
        return spec
    arch = getattr(model.config, "model_type", "")
    targets = TARGETS_BY_ARCH.get(arch)
    if targets is None:
        print(f"[warn] 未知架构 {arch!r}，回退到 target_modules='all-linear'")
        return "all-linear"
    return targets


def build_model(cfg: dict, vocab_size: int | None = None):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM

    m = cfg["model"]
    if m.get("_tiny_random"):
        # smoke 专用：本地随机初始化，绝不联网
        tc = LlamaConfig(vocab_size=int(vocab_size), hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=2048)
        return LlamaForCausalLM(tc)

    quant = m["quantization"]
    dtype = _torch_dtype(m["compute_dtype"]) if torch.cuda.is_available() else torch.float32
    kwargs: dict = {"attn_implementation": m["attn_implementation"]}
    kwargs[_dtype_kwarg()] = dtype

    if quant == "bnb_4bit":
        if not _bnb_available():
            raise SystemExit("quantization=bnb_4bit 需要 bitsandbytes，请先安装后重试")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=m.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(m.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": 0} if torch.cuda.is_available() else "auto"
    elif quant == "checkpoint_native":
        print("[warn] checkpoint_native 路径本地未验证（没有 31B 权重），"
              "请在 Kaggle L4x4 上做一次试加载确认。")
    elif quant != "none":
        raise SystemExit(f"未知的 quantization: {quant!r}")

    model = AutoModelForCausalLM.from_pretrained(m["id"], **kwargs)

    if quant == "checkpoint_native":
        qc = getattr(model.config, "quantization_config", None)
        if not qc:
            print("[warn] 期望权重自带量化配置，但模型上没有 quantization_config —— "
                  "确认你加载的确实是 w4a16 检查点，否则显存会按 bf16 结算。")
        else:
            print(f"[info] 使用检查点自带量化: {qc.get('quant_method', qc)}")
    return model


def attach_lora(model, cfg: dict):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    quantized = bool(getattr(model, "is_loaded_in_4bit", False)) or \
        getattr(model.config, "quantization_config", None) is not None
    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.config.use_cache = False

    lc = cfg["lora"]
    lcfg = LoraConfig(
        r=int(lc["r"]),
        lora_alpha=int(lc["alpha"]),
        lora_dropout=float(lc.get("dropout", 0.0)),
        target_modules=resolve_targets(model, lc["target_modules"]),
        bias=lc.get("bias", "none"),
        task_type=lc.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return model


def _prepare_for_training(model, cfg: dict) -> None:
    if cfg["training"].get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()


class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, items: list[dict]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def _warmup_kwargs(t: dict, n_samples: int, batch_effective: int, max_steps: int | None) -> dict:
    """transformers 5.x 删掉了 warmup_ratio，只保留 warmup_steps —— 这里做版本适配。"""
    from transformers import TrainingArguments

    ratio = float(t.get("warmup_ratio", 0.0))
    if ratio <= 0:
        return {"warmup_steps": int(t.get("warmup_steps", 0))}
    if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters:
        return {"warmup_ratio": ratio}
    if max_steps is not None:
        total = int(max_steps)
    else:
        per_epoch = max(1, math.ceil(n_samples / max(1, batch_effective)))
        total = per_epoch * max(1, int(math.ceil(float(t["epochs"]))))
    steps = int(total * ratio)
    print(f"[info] 当前 transformers 没有 warmup_ratio，"
          f"按 {ratio:g} x {total} 步换算 warmup_steps={steps}")
    return {"warmup_steps": steps}


def make_trainer(model, cfg: dict, features: list[dict], pad_token_id: int, max_steps: int | None):
    from transformers import Trainer, TrainingArguments

    t = cfg["training"]
    use_cuda = torch.cuda.is_available()
    batch_effective = int(t["per_device_batch_size"]) * int(t["grad_accum"])

    kwargs: dict = {
        "output_dir": cfg["output_dir"],
        "per_device_train_batch_size": int(t["per_device_batch_size"]),
        "gradient_accumulation_steps": int(t["grad_accum"]),
        "num_train_epochs": float(t["epochs"]),
        "max_steps": int(max_steps) if max_steps is not None else -1,
        "learning_rate": float(t["lr"]),
        "lr_scheduler_type": t.get("lr_scheduler", "cosine"),
        "weight_decay": float(t.get("weight_decay", 0.0)),
        "max_grad_norm": float(t.get("max_grad_norm", 1.0)),
        "optim": t.get("optim", "paged_adamw_8bit" if _bnb_available() else "adamw_torch"),
        "gradient_checkpointing": bool(t.get("gradient_checkpointing", False)),
        "logging_steps": int(t.get("logging_steps", 1)),
        "save_strategy": t.get("save_strategy", "no"),
        "bf16": use_cuda and cfg["model"].get("compute_dtype") == "bfloat16",
        "fp16": use_cuda and cfg["model"].get("compute_dtype") == "float16",
        "report_to": [],
        "seed": int(t.get("seed", 42)),
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
        "label_names": ["labels"],
    }
    if t.get("save_total_limit") is not None:
        kwargs["save_total_limit"] = int(t["save_total_limit"])
    kwargs.update(_warmup_kwargs(t, len(features), batch_effective, max_steps))

    # 版本自适应：不同 transformers 大版本的 TrainingArguments 参数集合会变，
    # 不认识的直接丢掉并**显式打印**，绝不静默忽略。
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    for key in sorted(k for k in kwargs if k not in accepted):
        print(f"[warn] 当前 transformers 不接受 TrainingArguments.{key}，已忽略")
    args = TrainingArguments(**{k: v for k, v in kwargs.items() if k in accepted})
    return Trainer(
        model=model,
        args=args,
        train_dataset=_ListDataset(features),
        data_collator=SftCollator(pad_token_id),
    )


def _mask_report(features: list[dict], pad_token_id: int) -> dict:
    """独立核对 mask 是否符合预期：非 assistant 位置必须全是 -100。"""
    IGNORE = -100
    total = sum(len(f["labels"]) for f in features)
    supervised = sum(1 for f in features for v in f["labels"] if v != IGNORE)
    head_is_ignored = all(f["labels"][0] == IGNORE for f in features)
    return {
        "tokens_total": total,
        "tokens_supervised": supervised,
        "supervised_ratio": round(supervised / total, 4) if total else 0.0,
        "head_ignored_for_all_samples": head_is_ignored,
        "pad_token_id": pad_token_id,
    }


def _save_adapter(model, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    return out_dir


def cmd_features(cfg: dict) -> int:
    samples = load_trajectories(cfg["data"]["path"])
    if cfg["model"].get("_tiny_random"):
        tok = StubTokenizer()
    else:
        tok = load_tokenizer(cfg["model"]["id"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    features, warnings = build_features(tok, samples, cfg["data"]["train_on"],
                                        int(cfg["data"]["max_seq_len"]))
    report = {"samples": len(samples), "mask": _mask_report(features, pad_id), "warnings": warnings}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_train(cfg: dict, max_steps: int | None) -> int:
    samples = load_trajectories(cfg["data"]["path"])
    tok = load_tokenizer(cfg["model"]["id"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    features, warnings = build_features(tok, samples, cfg["data"]["train_on"],
                                        int(cfg["data"]["max_seq_len"]))
    for w in warnings:
        print(f"[warn] {w}")

    print(f"[info] 训练样本 {len(features)} 条；mask 统计 {_mask_report(features, pad_id)}")
    model = build_model(cfg)
    model = attach_lora(model, cfg)
    _prepare_for_training(model, cfg)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer = make_trainer(model, cfg, features, pad_id, max_steps)
    result = trainer.train()

    out_dir = _save_adapter(model, Path(cfg["output_dir"]))
    summary = {
        "run_name": cfg.get("run_name"),
        "model_id": cfg["model"]["id"],
        "quantization": cfg["model"]["quantization"],
        "train_samples": len(features),
        "train_loss": round(float(result.training_loss), 4),
        "global_step": int(result.global_step),
        "adapter_dir": str(out_dir),
    }
    if torch.cuda.is_available():
        summary["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
        summary["gpu"] = torch.cuda.get_device_name(0)
    summary["adapter_check"] = inspect_adapter(out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    report_path = Path(cfg["output_dir"]) / "train_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def cmd_export(adapter_dir: str, submission_root: str, name: str) -> int:
    rep = export_adapter(adapter_dir, submission_root, name)
    scan = scan_extensions(submission_root)
    print(json.dumps({"adapter": rep, "submission_scan": scan}, ensure_ascii=False, indent=2))
    if not rep["ok"]:
        print("[fail] adapter 不满足提交要求", file=sys.stderr)
        return 1
    if not scan["ok"]:
        print("[fail] 提交根目录存在非白名单扩展名，整包会被拒", file=sys.stderr)
        return 1
    return 0


def cmd_smoke(work_dir: str, steps: int) -> int:
    """零下载端到端验证：不等任何模型下载，不占任何 GPU 配额。"""
    import tempfile

    root = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="gda_smoke_"))
    root.mkdir(parents=True, exist_ok=True)
    print(f"[smoke] 工作目录 {root}")

    tok = StubTokenizer()
    samples = make_synthetic_samples(4)
    features, warnings = build_features(tok, samples, "assistant_only", 512)
    assert not warnings, f"不该有 warning: {warnings}"

    mask = _mask_report(features, pad_token_id=0)
    assert mask["tokens_supervised"] > 0, "没有任何被监督的 token，mask 逻辑失效"
    assert mask["head_ignored_for_all_samples"], "序列开头（system 段）应被 mask"
    assert 0.01 < mask["supervised_ratio"] < 0.9, f"监督比例异常: {mask['supervised_ratio']}"
    print(f"[smoke] mask OK  {mask}")

    vocab_size = len(tok) + 8
    cfg = {
        "run_name": "smoke",
        "model": {"id": "tiny-random-llama", "quantization": "none", "_tiny_random": True,
                  "compute_dtype": "float32", "attn_implementation": "sdpa"},
        "lora": {"r": 8, "alpha": 16, "dropout": 0.0, "target_modules": "auto",
                 "bias": "none", "task_type": "CAUSAL_LM"},
        "data": {"train_on": "assistant_only", "max_seq_len": 512},
        "training": {"epochs": 1, "per_device_batch_size": 1, "grad_accum": 1,
                     "lr": 1e-3, "lr_scheduler": "constant", "warmup_ratio": 0.0,
                     "weight_decay": 0.0, "max_grad_norm": 1.0, "optim": "adamw_torch",
                     "gradient_checkpointing": False, "logging_steps": 1,
                     "save_strategy": "no", "seed": 42},
        "output_dir": str(root / "adapter"),
    }
    model = build_model(cfg, vocab_size=vocab_size)
    model = attach_lora(model, cfg)
    trainer = make_trainer(model, cfg, features, pad_token_id=0, max_steps=steps)
    result = trainer.train()
    assert result.global_step > 0, "训练没有推进"
    assert torch.isfinite(torch.tensor(result.training_loss)), "loss 不是有限值"

    adapter_dir = _save_adapter(model, Path(cfg["output_dir"]))
    check = inspect_adapter(adapter_dir)
    assert check["ok"], f"adapter 校验失败: {check['problems']}"
    print(f"[smoke] adapter OK  r={check['info']['r']} "
          f"targets={check['info']['target_modules']} {check['info']['weights_mb']}MB")

    sub_root = root / "submission"
    (sub_root / "skills" / "demo").mkdir(parents=True, exist_ok=True)
    (sub_root / "agent.yaml").write_text("name: smoke\n", encoding="utf-8")
    (sub_root / "skills" / "demo" / "SKILL.md").write_text("---\nname: demo\n---\n", encoding="utf-8")
    exp = export_adapter(adapter_dir, sub_root, "main_lora")
    assert exp["ok"], f"导出失败: {exp['problems']}"
    scan = scan_extensions(sub_root)
    assert scan["ok"], f"提交包扩展名违规: {scan['offenders']}"
    print(f"[smoke] 提交包 OK  {scan['by_extension']}")

    # 负例：塞一个 .sh 必须被门禁抓住
    (sub_root / "setup.sh").write_text("echo hi\n", encoding="utf-8")
    assert not scan_extensions(sub_root)["ok"], "非白名单扩展名应被拦截，但没有"
    (sub_root / "setup.sh").unlink()
    print("[smoke] 负例（.sh 混入）已被正确拦截")

    summary = {"ok": True, "steps": int(result.global_step),
               "loss": round(float(result.training_loss), 4), "mask": mask,
               "adapter": check["info"], "extensions": scan["by_extension"],
               "work_dir": str(root)}
    (root / "smoke_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="本地小模型 surrogate LoRA 训练器")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("smoke", help="零下载端到端验证整条流水线")
    p.add_argument("--work-dir", default="")
    p.add_argument("--steps", type=int, default=4)

    p = sub.add_parser("features", help="只做数据准备并打印 mask 统计")
    p.add_argument("--config", required=True)

    p = sub.add_parser("train", help="训练并保存 LoRA adapter")
    p.add_argument("--config", required=True)
    p.add_argument("--max-steps", type=int, default=None,
                   help="只跑 N 步（用于在真替身模型上快速验证显存是否够）")

    p = sub.add_parser("export", help="把 adapter 打进提交目录并校验")
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--submission-root", required=True)
    p.add_argument("--name", default="main_lora")

    args = ap.parse_args(argv)
    if args.cmd == "smoke":
        return cmd_smoke(args.work_dir, args.steps)
    if args.cmd == "features":
        return cmd_features(load_config(args.config))
    if args.cmd == "train":
        return cmd_train(load_config(args.config), args.max_steps)
    if args.cmd == "export":
        return cmd_export(args.adapter_dir, args.submission_root, args.name)
    raise SystemExit(f"未知子命令 {args.cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main())

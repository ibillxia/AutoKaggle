"""把训练好的 PEFT adapter 打包进比赛提交目录结构，并在提交前做硬校验。

规则依据（官方 overview「Model Selection and LoRA Adapters」）：
- adapter 必须是 PEFT 目录，内含 adapter_config.json + adapter_model.safetensors
- 放在 adapters/<adapter_name>/ 下
- 在 agent.yaml / sub_agents/*.yaml 里用 `adapter: <adapter_name>` 引用
- 提交 zip 只允许 .yaml .yml .md .txt .py .json .safetensors
  —— 出现别的扩展名（例如 .sh）会导致**整包被拒**
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ALLOWED_EXT = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}

# PEFT 保存的权重文件名，按优先级排列
WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")
CONFIG_NAME = "adapter_config.json"


def inspect_adapter(adapter_dir: str | Path) -> dict:
    """检查一个 adapter 目录是否满足提交要求，返回结构化结果（不抛异常）。"""
    d = Path(adapter_dir)
    result: dict = {"path": str(d), "exists": d.is_dir(), "problems": [], "info": {}}
    if not d.is_dir():
        result["problems"].append("目录不存在")
        return result

    cfg_path = d / CONFIG_NAME
    if not cfg_path.is_file():
        result["problems"].append(f"缺少 {CONFIG_NAME}")
    else:
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            result["problems"].append(f"{CONFIG_NAME} 不是合法 JSON: {exc}")
        else:
            result["info"]["peft_type"] = cfg.get("peft_type")
            result["info"]["r"] = cfg.get("r")
            result["info"]["lora_alpha"] = cfg.get("lora_alpha")
            result["info"]["target_modules"] = cfg.get("target_modules")
            result["info"]["base_model_name_or_path"] = cfg.get("base_model_name_or_path")
            if cfg.get("peft_type") != "LORA":
                result["problems"].append(f"peft_type 不是 LORA（是 {cfg.get('peft_type')!r}）")

    weights = [n for n in WEIGHT_NAMES if (d / n).is_file()]
    if not weights:
        result["problems"].append("缺少 adapter_model.safetensors（比赛要求 .safetensors）")
    else:
        result["info"]["weights"] = weights[0]
        if weights[0].endswith(".bin"):
            result["problems"].append(
                ".bin 权重不被接受：请用 model.save_pretrained(..., safe_serialization=True)")
        size = (d / weights[0]).stat().st_size
        result["info"]["weights_bytes"] = size
        result["info"]["weights_mb"] = round(size / 1024 / 1024, 2)

    junk = [p.name for p in d.iterdir()
            if p.is_file() and p.suffix and p.suffix not in ALLOWED_EXT]
    if junk:
        result["problems"].append(f"目录内有非白名单扩展名文件: {junk}")
    result["ok"] = not result["problems"]
    return result


def export(adapter_dir: str | Path, submission_root: str | Path, name: str) -> dict:
    """把 adapter 复制到 <submission_root>/adapters/<name>/，返回校验结果。"""
    src = Path(adapter_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"adapter 目录不存在: {src}")

    dst = Path(submission_root) / "adapters" / name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    copied = []
    for item in sorted(src.iterdir()):
        if item.is_file() and (item.suffix in ALLOWED_EXT or item.name == CONFIG_NAME):
            shutil.copy2(item, dst / item.name)
            copied.append(item.name)

    # 顺手清掉训练残留（optimizer states / 检查点目录），它们既不需要也可能超体积
    for junk_dir in ("checkpoint-0", "runs", "__pycache__"):
        stale = dst / junk_dir
        if stale.exists():
            shutil.rmtree(stale)

    report = inspect_adapter(dst)
    report["copied"] = copied
    report["dest"] = str(dst)
    return report


def scan_extensions(root: str | Path) -> dict:
    """扫描整个提交根目录，列出所有扩展名（用于赛前自检整包不会被拒）。"""
    root = Path(root)
    seen: dict[str, int] = {}
    offenders: list[str] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        seen[ext] = seen.get(ext, 0) + 1
        if ext and ext not in ALLOWED_EXT:
            offenders.append(str(p.relative_to(root)))
    return {"by_extension": dict(sorted(seen.items())),
            "offenders": sorted(offenders),
            "ok": not offenders}

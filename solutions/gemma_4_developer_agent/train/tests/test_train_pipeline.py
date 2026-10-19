"""surrogate 训练流水线的单元测试。

不联网、不下载、不需要 GPU：全部用 StubTokenizer + 构造的目录验证。
运行：  python -m pytest tests -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from adapter_export import export, inspect_adapter, scan_extensions  # noqa: E402
from sft_data import (  # noqa: E402
    IGNORE_INDEX,
    SftCollator,
    StubTokenizer,
    _as_ids,
    apply_mask,
    build_features,
    make_synthetic_samples,
    render,
)


class _FakeBatchEncoding:
    """模仿 transformers 5.x 的 BatchEncoding：不是 dict 子类，但有 .input_ids。

    4.x 时代它继承 dict，所以 `isinstance(x, dict)` 判断能过；5.x 不行。
    这个回归测试就是为了钉死这个差异。
    """

    def __init__(self, ids):
        self.input_ids = ids

    def __iter__(self):
        return iter(("input_ids", "attention_mask"))

    def __len__(self):
        return 2


class PickyTokenizer(StubTokenizer):
    """模拟「chat template 不认 tool role」的分词器，用于验证降级路径。"""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kw):
        for m in messages:
            if m["role"] not in ("system", "user", "assistant"):
                raise ValueError(f"unsupported role: {m['role']}")
        return super().apply_chat_template(messages, tokenize, add_generation_prompt, **kw)


# ---------------------------------------------------------------- sft_data

def test_as_ids_handles_all_transformer_return_shapes():
    """transformers 4.x 返回 list[int]，5.x 返回非 dict 的 BatchEncoding。"""
    assert _as_ids([1, 2, 3]) == [1, 2, 3]                       # 4.x
    assert _as_ids(_FakeBatchEncoding([4, 5])) == [4, 5]         # 5.x
    assert _as_ids({"input_ids": [6, 7]}) == [6, 7]              # 普通 dict
    assert _as_ids([[8, 9]]) == [8, 9]                           # 带 batch 维

    class _T:
        def __init__(self, v):
            self.v = v

        def tolist(self):
            return self.v

    assert _as_ids(_T([10, 11])) == [10, 11]                     # tensor 风格


def test_stub_tokenizer_is_prefix_consistent():
    """assistant 段定位依赖 chat template 的前缀一致性，这是最底层的前提。"""
    tok = StubTokenizer()
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "issue"},
        {"role": "assistant", "content": "abc"},
        {"role": "tool", "content": "result"},
        {"role": "assistant", "content": "def"},
    ]
    full, spans = render(tok, msgs)
    assert len(spans) == 2
    for (a, b), expected in zip(spans, ["abc", "def"]):
        decoded = "".join(tok._vocab[i] for i in full[a:b])
        assert expected in decoded


def test_apply_mask_ignores_everything_but_assistant():
    ids = list(range(20))
    labels = apply_mask(ids, [(5, 8), (12, 14)], "assistant_only")
    for i, v in enumerate(labels):
        if 5 <= i < 8 or 12 <= i < 14:
            assert v == ids[i]
        else:
            assert v == IGNORE_INDEX


def test_apply_mask_all_keeps_everything():
    ids = list(range(10))
    assert apply_mask(ids, [], "all") == ids


def test_apply_mask_rejects_unknown_mode():
    with pytest.raises(ValueError):
        apply_mask([1, 2, 3], [], "tool_only")


def test_features_supervise_only_assistant_and_warn_on_truncation():
    tok = StubTokenizer()
    features, warnings = build_features(tok, make_synthetic_samples(3), "assistant_only", 512)
    assert len(features) == 3
    assert not warnings
    for f in features:
        assert f["labels"][0] == IGNORE_INDEX          # system 段被 mask
        assert any(v != IGNORE_INDEX for v in f["labels"])
        assert len(f["input_ids"]) == len(f["labels"])

    long_features, long_warnings = build_features(tok, make_synthetic_samples(1),
                                                  "assistant_only", 40)
    assert len(long_features[0]["input_ids"]) == 40
    assert any("截断" in w for w in long_warnings)


def test_render_falls_back_when_template_rejects_tool_role():
    tok = PickyTokenizer()
    full, spans = render(tok, make_synthetic_samples(1)[0]["messages"])
    assert spans, "降级后仍应能定位 assistant 段"
    assert all(0 <= a < b <= len(full) for a, b in spans)


def test_collator_pads_labels_with_ignore_index():
    features = [
        {"id": "a", "input_ids": [1, 2, 3], "labels": [1, IGNORE_INDEX, 3]},
        {"id": "b", "input_ids": [4, 5], "labels": [4, 5]},
    ]
    batch = SftCollator(pad_token_id=0)(features)
    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"].tolist() == [[1, IGNORE_INDEX, 3], [4, 5, IGNORE_INDEX]]


# ---------------------------------------------------------- adapter_export

def _write_fake_adapter(d: Path, *, weights: str = "adapter_model.safetensors",
                        config: bool = True, peft_type: str = "LORA") -> Path:
    d.mkdir(parents=True, exist_ok=True)
    if config:
        (d / "adapter_config.json").write_text(json.dumps({
            "peft_type": peft_type, "r": 16, "lora_alpha": 32,
            "target_modules": ["q_proj"], "base_model_name_or_path": "x/y",
        }), encoding="utf-8")
    if weights:
        (d / weights).write_bytes(b"\x00" * 1024)
    return d


def test_inspect_adapter_accepts_valid_dir(tmp_path):
    d = _write_fake_adapter(tmp_path / "main_lora")
    rep = inspect_adapter(d)
    assert rep["ok"], rep["problems"]
    assert rep["info"]["r"] == 16
    assert rep["info"]["weights"] == "adapter_model.safetensors"


def test_inspect_adapter_rejects_bin_and_missing_config(tmp_path):
    d = _write_fake_adapter(tmp_path / "bin_only", weights="adapter_model.bin")
    rep = inspect_adapter(d)
    assert not rep["ok"]
    assert any(".bin" in p for p in rep["problems"])

    d2 = _write_fake_adapter(tmp_path / "no_cfg", config=False)
    assert not inspect_adapter(d2)["ok"]

    d3 = _write_fake_adapter(tmp_path / "not_lora", peft_type="PROMPT_TUNING")
    assert any("peft_type" in p for p in inspect_adapter(d3)["problems"])


def test_export_produces_expected_submission_layout(tmp_path):
    src = _write_fake_adapter(tmp_path / "src")
    root = tmp_path / "submission"
    rep = export(src, root, "main_lora")
    assert rep["ok"], rep["problems"]
    assert (root / "adapters" / "main_lora" / "adapter_config.json").is_file()
    assert (root / "adapters" / "main_lora" / "adapter_model.safetensors").is_file()


def test_scan_extensions_catches_non_whitelisted_file(tmp_path):
    (tmp_path / "agent.yaml").write_text("name: x\n", encoding="utf-8")
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "system.md").write_text("hi\n", encoding="utf-8")
    assert scan_extensions(tmp_path)["ok"]

    (tmp_path / "setup.sh").write_text("echo hi\n", encoding="utf-8")
    scan = scan_extensions(tmp_path)
    assert not scan["ok"]
    assert scan["offenders"] == ["setup.sh"]

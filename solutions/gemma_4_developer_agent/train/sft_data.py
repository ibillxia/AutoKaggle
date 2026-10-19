"""SWE-agent 轨迹 -> SFT 训练样本。

与比赛对齐的三点
----------------
1. 训练目标是 **agent 的工具调用行为**，不是普通问答。一条样本 = 一条多轮轨迹：
   user 报 issue -> assistant 调工具 -> tool 返回 -> ... -> submit_patch()。
2. 默认只对 assistant 段算 loss（``train_on: assistant_only``）。系统提示词是固定的、
   工具输出是环境给的，把它们的 token 也拿来算 loss 只会浪费容量，还会让模型学会复述日志。
3. 内置 ``StubTokenizer``，``smoke`` 模式**零下载**就能端到端验证
   「模板渲染 -> assistant 段定位 -> mask」这条链。
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import torch

IGNORE_INDEX = -100
SAFE_ROLES = ("system", "user", "assistant")


def _as_ids(out: Any) -> list[int]:
    """把不同 transformers 版本的返回值统一成 list[int]。

    4.x: ``apply_chat_template(tokenize=True)`` 直接返回 ``list[int]``。
    5.x: 返回 ``BatchEncoding`` —— 它**不再是 dict 子类**，但有 ``.input_ids``。
    这个差异在 4.x 上写出来的代码拿到 5.x 上会抛出
    ``invalid literal for int() with base 10: 'input_ids'``（因为遍历到了键名）。
    """
    if hasattr(out, "input_ids"):
        out = out.input_ids
    elif isinstance(out, Mapping) and "input_ids" in out:
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    return [int(x) for x in out]


class StubTokenizer:
    """零下载的字符级分词器，模板等价于 ChatML。

    唯一目的：让 smoke 测试真正走通「模板渲染 -> assistant 段定位 -> mask」这条链，
    而不必联网拉取任何分词器。**不要**拿它训练真实模型。
    """

    SPECIALS = ["<|bos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|tool|>", "<|end|>"]

    def __init__(self) -> None:
        self._vocab = list(self.SPECIALS)

    def __len__(self) -> int:
        return len(self._vocab)

    def _id(self, token: str) -> int:
        try:
            return self._vocab.index(token)
        except ValueError:
            self._vocab.append(token)
            return len(self._vocab) - 1

    def _encode(self, text: str) -> list[int]:
        return [self._id(ch) for ch in text]

    def __call__(self, text: str, add_special_tokens: bool = False, **kwargs) -> dict:
        return {"input_ids": self._encode(text)}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        if not tokenize:
            raise NotImplementedError("StubTokenizer 只支持 tokenize=True")
        ids = [self._id("<|bos|>")]
        for m in messages:
            ids.append(self._id("<|" + m["role"] + "|>"))
            ids.extend(self._encode(str(m.get("content") or "")))
            ids.append(self._id("<|end|>"))
        if add_generation_prompt:
            ids.append(self._id("<|assistant|>"))
        return ids


def load_tokenizer(model_id: str, **kwargs):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, **kwargs)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_trajectories(path: str | Path) -> list[dict]:
    path = Path(path)
    samples: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msgs = rec.get("messages")
            if not msgs:
                raise ValueError(f"{path}:{lineno} 缺少 messages 字段")
            samples.append({"id": rec.get("id", f"line{lineno}"), "messages": msgs})
    if not samples:
        raise ValueError(f"{path} 里没有任何样本")
    return samples


def make_synthetic_samples(n: int = 4) -> list[dict]:
    """smoke 用的合成轨迹：不看磁盘、不联网，形状与真实样本一致。"""
    system = "You are an autonomous software engineering agent. Call submit_patch() when done."
    out = []
    for k in range(n):
        out.append({
            "id": f"synthetic-{k:04d}",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Issue {k}: offset is wrong when page is zero."},
                {"role": "assistant", "content": "read_file(filepath=\"pkg/paginate.py\")"},
                {"role": "tool", "content": "1  def offset(page, size):\n2      return (page - 1) * size\n"},
                {"role": "assistant", "content": "run_command(command=\"python -m pytest tests -q -k pagination\")"},
                {"role": "tool", "content": "3 passed in 0.21s\n"},
                {"role": "assistant", "content": "edit_file(filepath=\"pkg/paginate.py\", old_string=\"(page - 1)\", new_string=\"(max(1, page) - 1)\")"},
                {"role": "tool", "content": "edited pkg/paginate.py"},
                {"role": "assistant", "content": "submit_patch()"},
            ],
        })
    return out


def _normalize_roles(messages: Sequence[dict]) -> list[dict]:
    """部分 chat template 不认 tool 等 role；降级为 user 并保留原 role 标记。"""
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content") or ""
        if role not in SAFE_ROLES:
            content = f"[{role}] {content}"
            role = "user"
        out.append({"role": role, "content": content})
    return out


def _render_strict(tok, msgs: list[dict]) -> tuple[list[int], list[tuple[int, int]]]:
    full = _as_ids(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False))
    spans: list[tuple[int, int]] = []
    cursor = 0
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        prefix = _as_ids(tok.apply_chat_template(msgs[:i], tokenize=True, add_generation_prompt=True))
        start = len(prefix)
        turn = _as_ids(tok.apply_chat_template(msgs[: i + 1], tokenize=True, add_generation_prompt=False))
        end = min(len(turn), len(full))
        if not (cursor <= start <= end):
            raise ValueError("chat template 非前缀一致，无法定位 assistant 段")
        spans.append((start, end))
        cursor = end
    return full, spans


def render(tok, messages: list[dict]) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        return _render_strict(tok, messages)
    except Exception:
        return _render_strict(tok, _normalize_roles(messages))


def apply_mask(input_ids: list[int], spans: list[tuple[int, int]], train_on: str) -> list[int]:
    labels = list(input_ids)
    if train_on == "all":
        return labels
    if train_on != "assistant_only":
        raise ValueError(f"未知的 train_on: {train_on!r}（可选 all / assistant_only）")
    keep = set()
    for a, b in spans:
        keep.update(range(a, b))
    for i in range(len(labels)):
        if i not in keep:
            labels[i] = IGNORE_INDEX
    return labels


def build_features(tok, samples: Sequence[dict], train_on: str, max_seq_len: int):
    """返回 (features, warnings)。features 为 [{'input_ids': [...], 'labels': [...], 'id': str}]。"""
    features, warnings = [], []
    for sample in samples:
        ids, spans = render(tok, sample["messages"])
        if train_on == "assistant_only" and not spans:
            warnings.append(f"{sample['id']}: 没定位到 assistant 段，退化为 all")
            labels = list(ids)
        else:
            labels = apply_mask(ids, spans, train_on)
        if len(ids) > max_seq_len:
            warnings.append(f"{sample['id']}: {len(ids)} -> {max_seq_len} tokens（截断）")
            ids, labels = ids[:max_seq_len], labels[:max_seq_len]
        features.append({"id": sample["id"], "input_ids": ids, "labels": labels})
    return features, warnings


class SftCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: Sequence[dict]) -> dict:
        width = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            ids, lab = list(f["input_ids"]), list(f["labels"])
            pad = width - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            labels.append(lab + [IGNORE_INDEX] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

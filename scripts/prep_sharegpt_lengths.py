#!/usr/bin/env python3
"""从 ShareGPT JSON 导出 (prompt_len, output_len) CSV，供 trace.kind=sharegpt 使用。

对齐 EcoServe `sample_sharegpt_requests`：
  - conversations >= 2；只取前两轮（human / assistant）
  - 过滤 prompt_len<4 / output_len<4 / prompt>max_seqlen / prompt+out>max_seqlen

分词：
  - 若已安装 transformers，用 --tokenizer（默认 hf-internal-testing/llama-tokenizer）
  - 否则默认 **char/4** 近似；可 --approx whitespace。对标 EcoServe 请装 transformers tokenizer

用法::
  .venv/bin/python scripts/prep_sharegpt_lengths.py \\
    --input input/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \\
    --output input/datasets/sharegpt_lengths_v1.csv \\
    --sample 4000 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


def _whitespace_len(text: str) -> int:
    return len(text.split()) if text else 0


def _char4_len(text: str) -> int:
    """粗近似 BPE：英文约 4 char/token；无 HF tokenizer 时默认。"""
    return max(1, len(text) // 4) if text else 0


def _make_counter(tokenizer_name: str | None, *, approx: str):
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer  # type: ignore

            tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

            def count(text: str) -> int:
                return len(tok(text, add_special_tokens=False).input_ids)

            return count, f"transformers:{tokenizer_name}"
        except Exception as e:  # noqa: BLE001
            print(f"[warn] tokenizer 不可用 ({e}); 回退 approx={approx}", file=sys.stderr)
    if approx == "whitespace":
        return _whitespace_len, "whitespace"
    return _char4_len, "char/4"


def iter_conversations(path: Path):
    """流式解析顶层 JSON 数组，避免整文件进内存。"""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        buf = ""
        # 读到 '['
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                raise SystemExit("空文件或非数组 JSON")
            buf += chunk
            i = 0
            while i < len(buf) and buf[i].isspace():
                i += 1
            if i < len(buf):
                if buf[i] != "[":
                    raise SystemExit("期望顶层 JSON 数组")
                buf = buf[i + 1 :]
                break
        while True:
            while True:
                j = 0
                while j < len(buf) and buf[j].isspace():
                    j += 1
                buf = buf[j:]
                if buf.startswith(","):
                    buf = buf[1:]
                    continue
                break
            if not buf:
                chunk = f.read(1 << 20)
                if not chunk:
                    return
                buf += chunk
                continue
            if buf[0] == "]":
                return
            while True:
                try:
                    obj, idx = decoder.raw_decode(buf)
                    break
                except json.JSONDecodeError:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        raise SystemExit("JSON 截断")
                    buf += chunk
            yield obj
            buf = buf[idx:]


def _role_value(turn: dict) -> tuple[str, str]:
    role = str(turn.get("from") or turn.get("role") or "").lower()
    val = turn.get("value") or turn.get("content") or ""
    return role, str(val)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=4000, help="写出行数（水库抽样）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seqlen", type=int, default=4096)
    ap.add_argument(
        "--tokenizer",
        default="",
        help="可选 HF tokenizer 名；空则用 --approx",
    )
    ap.add_argument(
        "--approx",
        choices=("char4", "whitespace"),
        default="char4",
        help="无 transformers 时的长度近似（默认 char/4，更接近 BPE）",
    )
    ap.add_argument(
        "--max-scan",
        type=int,
        default=0,
        help="最多扫描多少条对话（0=全部）；用于快速试跑",
    )
    args = ap.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"找不到输入: {args.input}")

    count_fn, mode = _make_counter(args.tokenizer or None, approx=args.approx)
    print(f"tokenizer_mode={mode} max_seqlen={args.max_seqlen} sample={args.sample}")

    rng = random.Random(args.seed)
    reservoir: list[tuple[int, int]] = []
    seen = kept_pre = 0
    for obj in iter_conversations(args.input):
        seen += 1
        if args.max_scan and seen > args.max_scan:
            break
        conv = obj.get("conversations") if isinstance(obj, dict) else None
        if not isinstance(conv, list) or len(conv) < 2:
            continue
        _r0, prompt = _role_value(conv[0])
        _r1, completion = _role_value(conv[1])
        prompt_len = count_fn(prompt)
        output_len = count_fn(completion)
        if prompt_len < 4 or output_len < 4:
            continue
        if prompt_len > args.max_seqlen or prompt_len + output_len > args.max_seqlen:
            continue
        kept_pre += 1
        item = (prompt_len, output_len)
        if len(reservoir) < args.sample:
            reservoir.append(item)
        else:
            j = rng.randrange(kept_pre)
            if j < args.sample:
                reservoir[j] = item
        if seen % 20000 == 0:
            print(f"  scanned={seen} kept_pool={kept_pre} reservoir={len(reservoir)}")

    if not reservoir:
        raise SystemExit("过滤后为空；检查输入或放宽 max_seqlen")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt_len", "output_len"])
        w.writerows(reservoir)

    pins = [a for a, _ in reservoir]
    pouts = [b for _, b in reservoir]
    print(
        f"wrote {args.output} n={len(reservoir)} "
        f"prompt_len[min/med/max]={min(pins)}/{sorted(pins)[len(pins)//2]}/{max(pins)} "
        f"output_len[min/med/max]={min(pouts)}/{sorted(pouts)[len(pouts)//2]}/{max(pouts)} "
        f"scanned={seen} passed_filter={kept_pre}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrated traj summary:
- combines "vs first" (reduction_vs_first, unique_visit_urls, redundant_visit_rate, shares)
- combines "bad call/mismatch" (bad_tool_calls, mismatch, parsed, other tools)
"""

import sys
import re
import json
import glob
import codecs
import argparse
from typing import Any, Dict, List, Tuple, Set

TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
ITER_NUM_RE = re.compile(r"(?:^|/|\\)iter(\d+)\.jsonl$", re.IGNORECASE)

def safe_div(a: float, b: float) -> float:
    return (a / b) if b else 0.0

def is_text(x: Any) -> bool:
    return isinstance(x, str)

def safe_strip(s: Any) -> str:
    if s is None:
        return ""
    if is_text(s):
        return s.strip()
    return str(s).strip()

def merge_name_counts(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v

def iter_sort_key(path: str) -> Tuple[int, str]:
    m = ITER_NUM_RE.search(path)
    if m:
        return (int(m.group(1)), path)
    return (10**9, path)

def expand_inputs(inputs: List[str]) -> List[str]:
    paths: List[str] = []
    for x in inputs:
        if any(ch in x for ch in ["*", "?", "[", "]"]):
            paths.extend(glob.glob(x))
        else:
            paths.append(x)
    paths = sorted(list(dict.fromkeys(paths)), key=iter_sort_key)
    return paths

def extract_urls_from_visit_args(args: Any) -> List[str]:
    """
    visit tool schema: {"url": [ ... ], "goal": "..."}
    but be robust to:
      - url as string
      - missing url
    """
    if not isinstance(args, dict):
        return []
    url_field = args.get("url")
    if url_field is None:
        return []
    if isinstance(url_field, list):
        out = []
        for u in url_field:
            if u is None:
                continue
            out.append(safe_strip(u))
        return [u for u in out if u]
    if is_text(url_field):
        u = safe_strip(url_field)
        return [u] if u else []
    return []

def count_tool_calls_in_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "tool_calls_total": 0,
        "tool_calls_parsed": 0,
        "bad_tool_calls": 0,

        "search_calls": 0,
        "visit_calls": 0,
        "other_tool_calls": 0,
        "other_tool_names": {},

        # visit url stats
        "visit_urls_total": 0,        # total url occurrences visited (sum of lengths of url arrays)
        "unique_visit_urls_set": set()  # type: Set[str]
    }

    msgs = rec.get("messages", [])
    if not isinstance(msgs, list):
        return out

    for m in msgs:
        if not isinstance(m, dict):
            continue
        content = m.get("content", "")
        if not is_text(content):
            continue

        blocks = TOOL_CALL_BLOCK_RE.findall(content)
        if not blocks:
            continue

        out["tool_calls_total"] += len(blocks)

        for blk in blocks:
            blk = safe_strip(blk)
            try:
                obj = json.loads(blk)
            except Exception:
                out["bad_tool_calls"] += 1
                continue

            out["tool_calls_parsed"] += 1
            name = obj.get("name", None)
            if name is None:
                out["bad_tool_calls"] += 1
                continue

            name = safe_strip(name)

            if name == "search":
                out["search_calls"] += 1
            elif name == "visit":
                out["visit_calls"] += 1
                args = obj.get("arguments", {})
                urls = extract_urls_from_visit_args(args)
                out["visit_urls_total"] += len(urls)
                for u in urls:
                    out["unique_visit_urls_set"].add(u)
            else:
                out["other_tool_calls"] += 1
                out["other_tool_names"][name] = out["other_tool_names"].get(name, 0) + 1

    return out

def count_file(path: str) -> Dict[str, Any]:
    samples = 0
    bad_lines = 0

    agg = {
        "tool_calls_total": 0,
        "tool_calls_parsed": 0,
        "bad_tool_calls": 0,
        "search_calls": 0,
        "visit_calls": 0,
        "other_tool_calls": 0,
        "other_tool_names": {},
        "visit_urls_total": 0,
        "unique_visit_urls_set": set(),  # type: Set[str]
    }

    with codecs.open(path, "r", "utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples += 1
            try:
                rec = json.loads(line)
            except Exception:
                bad_lines += 1
                continue
            if not isinstance(rec, dict):
                continue

            r = count_tool_calls_in_record(rec)
            agg["tool_calls_total"] += r["tool_calls_total"]
            agg["tool_calls_parsed"] += r["tool_calls_parsed"]
            agg["bad_tool_calls"] += r["bad_tool_calls"]
            agg["search_calls"] += r["search_calls"]
            agg["visit_calls"] += r["visit_calls"]
            agg["other_tool_calls"] += r["other_tool_calls"]
            merge_name_counts(agg["other_tool_names"], r["other_tool_names"])
            agg["visit_urls_total"] += r["visit_urls_total"]
            agg["unique_visit_urls_set"].update(r["unique_visit_urls_set"])

    tool_calls_total = float(agg["tool_calls_total"])
    search_calls = float(agg["search_calls"])
    visit_calls = float(agg["visit_calls"])
    other_calls = float(agg["other_tool_calls"])
    bad_tool_calls = float(agg["bad_tool_calls"])
    parsed = float(agg["tool_calls_parsed"])

    mismatch = int(agg["tool_calls_total"] - (agg["search_calls"] + agg["visit_calls"] + agg["other_tool_calls"]))
    # 上面 mismatch 一般等于 bad_tool_calls，但保留两者，方便你对齐口径

    per_sample = safe_div(tool_calls_total, float(samples))
    search_share = safe_div(search_calls, tool_calls_total)
    visit_share = safe_div(visit_calls, tool_calls_total)

    unique_visit_urls = len(agg["unique_visit_urls_set"])
    visit_urls_total = float(agg["visit_urls_total"])  # 注意这是 url occurrence，不是 visit_calls
    redundant_visit_rate = 0.0
    if visit_urls_total > 0:
        redundant_visit_rate = 1.0 - safe_div(float(unique_visit_urls), visit_urls_total)

    summary = {
        "path": path,
        "samples": samples,
        "bad_lines": bad_lines,

        # tool calls
        "tool_calls_total": int(tool_calls_total),
        "tool_calls_parsed": int(parsed),
        "bad_tool_calls": int(bad_tool_calls),
        "mismatch": mismatch,

        "search_calls": int(search_calls),
        "visit_calls": int(visit_calls),
        "other_tool_calls": int(other_calls),
        "other_tool_names": agg["other_tool_names"],

        "tool_calls_per_sample": per_sample,
        "search_share": search_share,
        "visit_share": visit_share,

        # visit url redundancy
        "unique_visit_urls": unique_visit_urls,
        "visit_urls_total": int(visit_urls_total),
        "redundant_visit_rate": redundant_visit_rate,

        # placeholder (filled later)
        "reduction_vs_first": 0.0,
    }
    return summary

def add_reduction_vs_first(summaries: List[Dict[str, Any]]) -> None:
    if not summaries:
        return
    base_total = float(summaries[0]["tool_calls_total"])
    for s in summaries:
        curr_total = float(s["tool_calls_total"])
        s["reduction_vs_first"] = safe_div(base_total - curr_total, base_total)

def write_out(out_path: str, records: List[Dict[str, Any]]) -> None:
    if not out_path:
        return
    lower = out_path.lower()
    if lower.endswith(".jsonl"):
        with codecs.open(out_path, "w", "utf-8") as fo:
            for r in records:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        with codecs.open(out_path, "w", "utf-8") as fo:
            fo.write(json.dumps(records, ensure_ascii=False, indent=2) + "\n")

def print_line(s: Dict[str, Any]) -> None:
    # 一行里把你两种想看的都放进去
    print(
        f"{s['path']}\t"
        f"samples={s['samples']}\t"
        f"tool_calls_total={s['tool_calls_total']}\t"
        f"search={s['search_calls']}\t"
        f"visit={s['visit_calls']}\t"
        f"other={s['other_tool_calls']}\t"
        f"bad_tool_calls={s['bad_tool_calls']}\t"
        f"mismatch={s['mismatch']}\t"
        f"per_sample={s['tool_calls_per_sample']:.3f}\t"
        f"search_share={s['search_share']:.3f}\t"
        f"visit_share={s['visit_share']:.3f}\t"
        f"unique_visit_urls={s['unique_visit_urls']}\t"
        f"visit_urls_total={s['visit_urls_total']}\t"
        f"redundant_visit_rate={s['redundant_visit_rate']:.3f}\t"
        f"reduction_vs_first={s['reduction_vs_first']:.3f}"
    )

def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize tool trajectories (integrated metrics).")
    ap.add_argument("files", nargs="+", help="jsonl file paths or glob patterns (e.g., iter*.jsonl)")
    ap.add_argument("--out", default="", help="write results to .json or .jsonl")
    ap.add_argument("--show-other", action="store_true", help="print other tool name counts")
    args = ap.parse_args()

    paths = expand_inputs(args.files)
    if not paths:
        print("No input files matched.", file=sys.stderr)
        return 2

    summaries = [count_file(p) for p in paths]
    add_reduction_vs_first(summaries)

    for s in summaries:
        print_line(s)
        if args.show_other and s.get("other_tool_names"):
            items = sorted(s["other_tool_names"].items(), key=lambda kv: (-kv[1], kv[0]))
            for name, cnt in items:
                print(f"  other_tool\t{name}\t{cnt}")

    if args.out:
        write_out(args.out, summaries)

    return 0

if __name__ == "__main__":
    sys.exit(main())

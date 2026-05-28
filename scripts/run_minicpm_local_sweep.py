#!/usr/bin/env python3
"""
run_minicpm_local_sweep.py
──────────────────────────
Full MVTec AD zero-shot anomaly detection sweep with MiniCPM-V 4.6 Q4_K_M
running locally via llama-cpp-python (CPU, no cloud, no API keys).

Uses the **direct mtmd API** (mtmd_helper_eval_chunks + sample loop) instead
of create_chat_completion.  This fixes the image-blind bug in Llava15ChatHandler
where llama.eval(prompt_tokens) inside generate() destroys image embeddings by
calling kv_cache_seq_rm before decoding.

AUROC scoring: confidence [0.0–1.0] used directly as the anomaly score,
same convention as the Gemini/Claude cloud backends.

Usage:
    python scripts/run_minicpm_local_sweep.py [--categories bottle capsule ...]
                                               [--limit N]
                                               [--threads T]
                                               [--max-tokens N]
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT        = Path(__file__).resolve().parent.parent
MODEL_PATH  = ROOT / "models/minicpm-v/MiniCPM-V-4_6-Q4_K_M.gguf"
MMPROJ_PATH = ROOT / "models/minicpm-v/mmproj-model-f16.gguf"
MVTEC_DIR   = ROOT / "data/mvtec"
RESULTS_DIR = ROOT / "results"

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile",
    "toothbrush", "transistor", "wood", "zipper",
]

# ── Prompt ────────────────────────────────────────────────────────────────────
# Describe-then-JSON: forces visual grounding before confidence scoring.
# confidence = P(defective) ∈ [0,1]. Same semantics as Gemini/Claude backends.
SYSTEM_PROMPT = (
    "You are a precision industrial quality inspector specialising in visual "
    "anomaly detection of manufactured components. "
    "The 'confidence' score must express the PROBABILITY this component IS DEFECTIVE: "
    "0.0 = certainly normal, 1.0 = certainly defective. "
    "A clean uniform surface MUST score near 0.0; obvious damage MUST score near 1.0."
)

# Prompt C_calibrated_scale — best from smoke test (gap=0.647 vs 0.000 for A, nan for B)
USER_PROMPT = """Rate the defect probability for this manufactured component image.

Scale reference:
  0.02 = smooth uniform surface, no issues whatsoever
  0.20 = minor natural variation, within normal tolerance
  0.50 = ambiguous — possible very subtle defect
  0.80 = likely defective — probable defect visible
  0.97 = clearly defective — obvious crack / hole / damage

Respond with ONLY valid JSON (no markdown, no extra text):
{"is_anomalous": <true/false>, "confidence": <float 0.0-1.0>, "defect_type": "<crack|scratch|contamination|hole|dent|discoloration|missing|none>", "description": "<surface observation>"}"""


# ── JSON parsing ──────────────────────────────────────────────────────────────
def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


def parse_json_response(text: str) -> tuple[dict[str, Any], bool]:
    t = _strip_thinking(text).strip()

    def _try(s: str) -> dict | None:
        try:
            return json.loads(s)
        except Exception:
            return None

    if d := _try(t):
        return _normalise(d), False

    if block := _extract_braces(t):
        if d := _try(block):
            return _normalise(d), False

    stripped = re.sub(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", r"\1", t).strip()
    if stripped != t:
        if d := _try(stripped):
            return _normalise(d), False
        if block := _extract_braces(stripped):
            if d := _try(block):
                return _normalise(d), False

    if d := _regex_extract(t):
        return _normalise(d), False

    return {}, True


def _extract_braces(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, prev = 0, False, ""
    for i, ch in enumerate(text[start:], start):
        if ch == '"' and prev != "\\":
            in_str = not in_str
        if not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        prev = ch
    return None


def _regex_extract(text: str) -> dict | None:
    # Handle both quoted JSON keys and plain text format (e.g. "confidence: 0.2")
    am = re.search(r'"?[Ii]s_anomalous"?\s*:\s*(true|false|"true"|"false"|"yes"|"no"|1|0)', text, re.I)
    cm = re.search(r'"?[Cc]onfidence"?\s*:\s*([0-9]*\.?[0-9]+)', text)
    if not cm:
        return None
    d: dict[str, Any] = {}
    d["confidence"] = max(0.0, min(1.0, float(cm.group(1))))
    if am:
        raw = am.group(1).strip('"').lower()
        d["is_anomalous"] = raw in ("true", "yes", "1")
    else:
        d["is_anomalous"] = d["confidence"] >= 0.5
    if dm := re.search(r'"?[Dd]escription"?\s*:\s*"?([^",\n}{]{3,80})', text):
        d["description"] = dm.group(1).strip('" ')
    if dt := re.search(r'"?[Dd]efect_type"?\s*:\s*"?([^",\n}{]{2,30})', text):
        d["defect_type"] = dt.group(1).strip('" ')
    return d


def _normalise(raw: dict) -> dict:
    out: dict[str, Any] = {}
    v = raw.get("is_anomalous")
    if isinstance(v, bool):
        out["is_anomalous"] = v
    elif isinstance(v, str):
        out["is_anomalous"] = v.lower() in ("true", "yes", "1")
    elif isinstance(v, int):
        out["is_anomalous"] = v != 0
    else:
        return {}
    conf = raw.get("confidence", 1.0 if out["is_anomalous"] else 0.0)
    try:
        out["confidence"] = max(0.0, min(1.0, float(conf) if conf is not None else 0.0))
    except (TypeError, ValueError):
        out["confidence"] = 1.0 if out["is_anomalous"] else 0.0
    for key in ("defect_type", "description", "region"):
        if v := raw.get(key):
            out[key] = str(v)
    return out


# ── Direct mtmd inference ─────────────────────────────────────────────────────
def _jpeg_bytes(img_path: Path, max_side: int = 448) -> bytes:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def infer_image(
    llm: Any,
    mtmd_ctx: Any,
    img_path: Path,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 300,
) -> str:
    """Direct mtmd inference: per-chunk eval → sample loop.

    Uses mtmd_helper_eval_chunk_single (not eval_chunks) because eval_chunks
    returns -1 on the 2nd+ call within the same mtmd_ctx due to internal
    vision-encoder state not being reset by llm.reset().

    Text chunks → llama.eval() (properly updates n_tokens + KV cache)
    Image chunks → mtmd_helper_eval_chunk_single (writes vision embeddings)
    Generation → llm.sample() + llm.eval([tok]) loop (no create_completion)
    """
    import llama_cpp as _lc
    from llama_cpp import mtmd_cpp

    marker = mtmd_cpp.mtmd_default_marker().decode("utf-8")
    prompt_text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{marker}\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    jpeg = _jpeg_bytes(img_path)
    n = len(jpeg)
    arr = (ctypes.c_uint8 * n).from_buffer_copy(jpeg)
    bitmap = mtmd_cpp.mtmd_helper_bitmap_init_from_buf(mtmd_ctx, arr, n)
    if bitmap is None:
        raise RuntimeError(f"bitmap init failed: {img_path.name}")

    try:
        inp = mtmd_cpp.mtmd_input_text()
        inp.text = prompt_text.encode("utf-8")
        inp.add_special = True
        inp.parse_special = True

        chunks = mtmd_cpp.mtmd_input_chunks_init()
        if chunks is None:
            raise RuntimeError("mtmd_input_chunks_init failed")
        try:
            bmp_arr = (mtmd_cpp.mtmd_bitmap_p_ctypes * 1)(bitmap)
            ret = mtmd_cpp.mtmd_tokenize(mtmd_ctx, chunks, ctypes.byref(inp), bmp_arr, 1)
            if ret != 0:
                raise RuntimeError(f"mtmd_tokenize error {ret}")

            n_chunks = mtmd_cpp.mtmd_input_chunks_size(chunks)
            if n_chunks == 0:
                raise RuntimeError("mtmd_tokenize: 0 chunks")

            llm.reset()

            for ci in range(n_chunks):
                chunk = mtmd_cpp.mtmd_input_chunks_get(chunks, ci)
                if chunk is None:
                    continue
                ctype = mtmd_cpp.mtmd_input_chunk_get_type(chunk)

                if ctype == mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_TEXT:
                    n_out = ctypes.c_size_t(0)
                    tptr = mtmd_cpp.mtmd_input_chunk_get_tokens_text(
                        chunk, ctypes.byref(n_out)
                    )
                    if tptr is not None and n_out.value > 0:
                        toks = [tptr[j] for j in range(n_out.value)]
                        llm.eval(toks)

                elif ctype in (
                    mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_IMAGE,
                    mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_AUDIO,
                ):
                    new_n_past = _lc.llama_pos(0)
                    ret = mtmd_cpp.mtmd_helper_eval_chunk_single(
                        mtmd_ctx,
                        llm._ctx.ctx,
                        chunk,
                        _lc.llama_pos(llm.n_tokens),
                        _lc.llama_seq_id(0),
                        llm.n_batch,
                        False,
                        ctypes.byref(new_n_past),
                    )
                    if ret != 0:
                        raise RuntimeError(
                            f"mtmd_helper_eval_chunk_single error {ret} chunk {ci}"
                        )
                    llm.n_tokens = new_n_past.value

        finally:
            mtmd_cpp.mtmd_input_chunks_free(chunks)
    finally:
        mtmd_cpp.mtmd_bitmap_free(bitmap)

    llm._sampler = llm._init_sampler(
        top_k=1,
        top_p=1.0,
        min_p=0.0,
        temp=1e-6,
        repeat_penalty=1.05,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        tfs_z=1.0,
        mirostat_mode=0,
        penalize_nl=True,
    )

    eos = llm.token_eos()
    tokens_out: list[int] = []
    while len(tokens_out) < max_tokens:
        tok = llm.sample()
        if tok == eos:
            break
        tokens_out.append(tok)
        llm.eval([tok])

    return llm.detokenize(tokens_out).decode("utf-8", errors="replace")


# ── Metrics ───────────────────────────────────────────────────────────────────
def auroc(y_true: list[int], y_score: list[float]) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def f1_at_threshold(y_true: list[int], y_score: list[float], thr: float = 0.5) -> float:
    try:
        from sklearn.metrics import f1_score
        y_pred = [1 if s >= thr else 0 for s in y_score]
        return float(f1_score(y_true, y_pred, zero_division=0))
    except Exception:
        return float("nan")


# ── Dataset helpers ───────────────────────────────────────────────────────────
def collect_images(category: str, limit: int | None) -> list[tuple[Path, int]]:
    cat_dir = MVTEC_DIR / category / "test"
    items: list[tuple[Path, int]] = []
    for split_dir in sorted(cat_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        label = 0 if split_dir.name == "good" else 1
        for img_path in sorted(split_dir.glob("*.png")):
            items.append((img_path, label))
    if limit:
        good  = [(p, l) for p, l in items if l == 0][: limit // 2]
        bad   = [(p, l) for p, l in items if l == 1][: limit - len(good)]
        items = sorted(good + bad, key=lambda x: x[0])
    return items


MODEL_ID = "edge/minicpm-v-4.6-q4km"
BACKEND  = "llama_cpp"


def result_path(category: str) -> Path:
    run_id = hashlib.md5(f"minicpm46_{category}".encode()).hexdigest()[:8]
    return RESULTS_DIR / f"{run_id}_mvtec_{category}.jsonl"


def _load_category_results(path: Path, out: dict, category: str) -> None:
    y_true, y_score, times = [], [], []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                y_true.append(r["sample_label"])
                pred = r.get("prediction", {})
                y_score.append(pred.get("confidence", 0.5))
                times.append(pred.get("latency_ms", 0))
            except Exception:
                pass
    if y_true:
        out[category] = {
            "auroc": auroc(y_true, y_score),
            "f1": f1_at_threshold(y_true, y_score),
            "accuracy": sum(
                1 for yt, yp in zip(y_true, [s >= 0.5 for s in y_score]) if (yt == 1) == yp
            ) / len(y_true),
            "n": len(y_true),
            "n_errors": 0,
            "avg_ms": float(np.mean(times)) if times else 0.0,
        }


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="MiniCPM-V 4.6 local MVTec sweep (direct mtmd)")
    ap.add_argument("--categories", nargs="+", default=ALL_CATEGORIES)
    ap.add_argument("--limit",      type=int, default=None,
                    help="Max images per category (balanced). Default: all.")
    ap.add_argument("--threads",    type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=300)
    args = ap.parse_args()

    for p, name in [(MODEL_PATH, "model GGUF"), (MMPROJ_PATH, "mmproj GGUF"), (MVTEC_DIR, "MVTec data")]:
        if not p.exists():
            print(f"[ERROR] {name} not found: {p}", file=sys.stderr)
            sys.exit(1)

    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  MiniCPM-V 4.6 Q4_K_M — MVTec AD Zero-Shot Sweep (direct mtmd)")
    print(f"  Model    : {MODEL_PATH.name}")
    print(f"  Mmproj   : {MMPROJ_PATH.name}")
    print(f"  Device   : Intel CPU ({args.threads} threads)")
    print(f"  Prompt   : C_calibrated_scale (gap=0.647 on smoke test)")
    print(f"  Cats     : {args.categories}")
    print(f"  Limit    : {args.limit or 'all'} images/category")
    print(f"{'='*70}\n")

    import llama_cpp as _lc
    from llama_cpp import Llama, mtmd_cpp

    print("[load] Loading language model…")
    t0 = time.time()
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=8192,
        n_batch=512,
        n_threads=args.threads,
        n_gpu_layers=0,
        verbose=False,
    )
    print(f"[load] LM ready in {time.time()-t0:.1f}s")

    print("[load] Initialising vision encoder (mtmd)…")
    t0 = time.time()
    ctx_params = mtmd_cpp.mtmd_context_params_default()
    ctx_params.use_gpu = False
    ctx_params.n_threads = args.threads
    mtmd_ctx = mtmd_cpp.mtmd_init_from_file(str(MMPROJ_PATH).encode(), llm.model, ctx_params)
    if mtmd_ctx is None:
        print("[ERROR] mtmd_init_from_file failed", file=sys.stderr)
        sys.exit(1)
    print(f"[load] mtmd ready in {time.time()-t0:.1f}s\n")

    category_results: dict[str, dict] = {}

    for category in args.categories:
        images   = collect_images(category, args.limit)
        out_path = result_path(category)

        done_paths: set[str] = set()
        if out_path.exists():
            with open(out_path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        img = r.get("prediction", {}).get("image_path") or r.get("image_path", "")
                        done_paths.add(img)
                    except Exception:
                        pass
            if len(done_paths) >= len(images):
                print(f"[{category}] Already complete ({len(done_paths)} images) — skipping.")
                _load_category_results(out_path, category_results, category)
                continue

        print(f"\n{'─'*60}")
        print(f"  Category: {category}  ({len(images)} images, {len(done_paths)} done)")
        print(f"  Output  : {out_path.name}")
        print(f"{'─'*60}")

        y_true: list[int]    = []
        y_score: list[float] = []
        cat_times: list[float] = []
        n_errors = 0

        with open(out_path, "a") as fout:
            for idx, (img_path, label) in enumerate(images):
                rel = str(img_path.relative_to(ROOT))
                if rel in done_paths:
                    continue

                t_start = time.time()
                try:
                    raw_text   = infer_image(llm, mtmd_ctx, img_path,
                                             SYSTEM_PROMPT, USER_PROMPT,
                                             max_tokens=args.max_tokens)
                    latency_ms = (time.time() - t_start) * 1000

                    parsed, parse_error = parse_json_response(raw_text)

                    pred_anomalous = parsed.get("is_anomalous", False)
                    confidence     = parsed.get("confidence", 1.0 if pred_anomalous else 0.0)

                    record = {
                        "experiment_id": f"minicpm46_mvtec_{category}",
                        "model_id":      MODEL_ID,
                        "backend":       BACKEND,
                        "dataset":       "mvtec",
                        "category":      category,
                        "sample_label":  label,
                        "prediction": {
                            "image_path":   rel,
                            "defect_type":  img_path.parent.name,
                            "is_anomalous": pred_anomalous,
                            "confidence":   confidence,
                            "description":  parsed.get("description", ""),
                            "defect_type_pred": parsed.get("defect_type", ""),
                            "raw_response": raw_text,
                            "latency_ms":   round(latency_ms, 1),
                            "cost_usd":     0.0,
                            "tokens_in":    0,
                            "tokens_out":   0,
                            "parse_error":  parse_error,
                        },
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    fout.write(json.dumps(record) + "\n")
                    fout.flush()

                    y_true.append(label)
                    y_score.append(confidence)
                    cat_times.append(latency_ms)
                    if parse_error:
                        n_errors += 1

                    correct = pred_anomalous == (label == 1)
                    mark    = "✓" if correct else "✗"
                    gt      = "ANOM" if label == 1 else "GOOD"
                    pred_s  = "ANOM" if pred_anomalous else "GOOD"
                    print(
                        f"  [{idx+1:3d}/{len(images)}] {mark} gt={gt} pred={pred_s} "
                        f"conf={confidence:.2f} | {img_path.parent.name:20s} "
                        f"| {latency_ms/1000:.1f}s"
                    )

                except Exception as exc:
                    latency_ms = (time.time() - t_start) * 1000
                    print(f"  [{idx+1:3d}/{len(images)}] ERROR: {exc}")
                    record = {
                        "experiment_id": f"minicpm46_mvtec_{category}",
                        "model_id":      MODEL_ID,
                        "backend":       BACKEND,
                        "dataset":       "mvtec",
                        "category":      category,
                        "sample_label":  label,
                        "prediction": {
                            "image_path":   rel,
                            "defect_type":  img_path.parent.name,
                            "is_anomalous": False,
                            "confidence":   0.5,
                            "raw_response": str(exc),
                            "latency_ms":   round(latency_ms, 1),
                            "cost_usd":     0.0,
                            "tokens_in":    0,
                            "tokens_out":   0,
                            "parse_error":  True,
                            "error":        str(exc),
                        },
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    fout.write(json.dumps(record) + "\n")
                    fout.flush()
                    y_true.append(label)
                    y_score.append(0.5)
                    n_errors += 1

        auc   = auroc(y_true, y_score)
        f1    = f1_at_threshold(y_true, y_score)
        acc   = sum(
            1 for yt, yp in zip(y_true, [s >= 0.5 for s in y_score]) if (yt == 1) == yp
        ) / max(len(y_true), 1)
        avg_ms = float(np.mean(cat_times)) if cat_times else 0.0

        category_results[category] = {
            "auroc": auc, "f1": f1, "accuracy": acc,
            "n": len(y_true), "n_errors": n_errors, "avg_ms": avg_ms,
        }

        print(f"\n  ┌─ {category} summary ───────────────────────────────────")
        print(f"  │  AUROC : {auc:.4f}")
        print(f"  │  F1    : {f1:.4f}  |  Acc: {acc:.4f}")
        print(f"  │  Avg   : {avg_ms/1000:.1f}s/image  |  Errors: {n_errors}/{len(y_true)}")
        print(f"  └{'─'*50}")

    # ── Final leaderboard ─────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  MiniCPM-V 4.6 Q4_K_M — MVTec Full Sweep Results")
    print(f"  {'Category':<15} {'AUROC':>7} {'F1':>7} {'Acc':>7} {'N':>5} {'ms/img':>8}")
    print(f"  {'─'*55}")

    aurocs: list[float] = []
    for cat in args.categories:
        r = category_results.get(cat)
        if r is None:
            print(f"  {cat:<15}  (no data)")
            continue
        auc_s = f"{r['auroc']:.4f}" if not np.isnan(r["auroc"]) else "  n/a "
        f1_s  = f"{r['f1']:.4f}"    if not np.isnan(r["f1"])    else "  n/a "
        print(f"  {cat:<15} {auc_s:>7} {f1_s:>7} {r['accuracy']:>7.4f}"
              f" {r['n']:>5} {r['avg_ms']:>7.0f}")
        if not np.isnan(r["auroc"]):
            aurocs.append(r["auroc"])

    if aurocs:
        print(f"  {'─'*55}")
        print(f"  {'MEAN':<15} {np.mean(aurocs):>7.4f}")

    print(f"{'='*70}\n")
    print(f"Results saved to: {RESULTS_DIR}/")

    mtmd_cpp.mtmd_free(mtmd_ctx)


if __name__ == "__main__":
    main()

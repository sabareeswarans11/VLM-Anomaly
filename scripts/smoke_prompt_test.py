#!/usr/bin/env python3
"""
smoke_prompt_test.py  (v3 — direct mtmd API)
─────────────────────────────────────────────
Smoke-test 3 prompt variants on 6 diverse MVTec images using the
**direct mtmd API** (bypasses create_chat_completion).

Root cause of the image-blind bug in v1/v2:
  Llava15ChatHandler.create_chat_completion → llama.generate(prompt_tokens)
  → llama.eval(tokens) calls kv_cache_seq_rm BEFORE the batch decode.
  This removes image embeddings written by mtmd_helper_eval_chunk_single,
  so the model never sees the image and gives identical text hallucinations.

Fix (this script):
  mtmd_helper_eval_chunks → llm.n_tokens = new_n_past
  → llm.sample() loop with reset=False (image embeddings stay in KV cache).

Metrics: AUROC gap = avg_conf(anomalous) − avg_conf(normal); higher = better.
Target: gap ≥ 0.3 for the sweep to reach AUROC ≥ 0.60.

Usage:
    python scripts/smoke_prompt_test.py [--max-tokens 300]
"""
from __future__ import annotations

import argparse
import ctypes
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

ROOT        = Path(__file__).resolve().parent.parent
MODEL_PATH  = ROOT / "models/minicpm-v/MiniCPM-V-4_6-Q4_K_M.gguf"
MMPROJ_PATH = ROOT / "models/minicpm-v/mmproj-model-f16.gguf"
MVTEC       = ROOT / "data/mvtec"

# 6 diverse images: 3 normal + 3 anomalous across 3 categories
TEST_IMAGES = [
    (MVTEC / "bottle/test/good/000.png",         0, "bottle-normal"),
    (MVTEC / "bottle/test/broken_large/000.png", 1, "bottle-broken"),
    (MVTEC / "carpet/test/good/000.png",          0, "carpet-normal"),
    (MVTEC / "carpet/test/cut/000.png",           1, "carpet-cut"),
    (MVTEC / "metal_nut/test/good/000.png",       0, "metalnut-normal"),
    (MVTEC / "metal_nut/test/scratch/000.png",    1, "metalnut-scratch"),
]

# ── Prompt variants ───────────────────────────────────────────────────────────
_SYS = (
    "You are a precision industrial quality inspector specialising in visual "
    "anomaly detection of manufactured components. "
    "The 'confidence' score must express the PROBABILITY this component IS DEFECTIVE: "
    "0.0 = certainly normal, 1.0 = certainly defective."
)

PROMPTS: dict[str, tuple[str, str]] = {
    # Variant A: JSON-first with examples (forces exact format).
    "A_json_with_examples": (
        _SYS,
        """Inspect this manufactured component image for defects.

Respond with ONLY a JSON object — no markdown, no text before or after:
{"is_anomalous": <true or false>, "confidence": <float 0.0-1.0>, "defect_type": "<defect type or none>", "description": "<one sentence about the surface>"}

Examples:
  Normal: {"is_anomalous": false, "confidence": 0.05, "defect_type": "none", "description": "clean uniform surface with consistent texture"}
  Defective: {"is_anomalous": true, "confidence": 0.93, "defect_type": "crack", "description": "clear crack visible across the surface"}

confidence = probability this IS defective. 0.0=certainly normal, 1.0=certainly defective.""",
    ),

    # Variant B: describe-then-JSON (forces visual grounding before scoring).
    "B_describe_then_json": (
        _SYS,
        """Examine this manufactured component image carefully.

Step 1: Describe in one sentence what you see — texture, colour, surface condition,
any marks, cracks, holes, stains, or deformations.

Step 2: Output ONLY this JSON on the next line (no markdown):
{"is_anomalous": <true/false>, "confidence": <0.0-1.0 prob IS defective>, "defect_type": "<crack|scratch|contamination|hole|dent|discoloration|missing|none>", "description": "<your step-1 observation>"}

Calibration: 0.0-0.1=clearly normal, 0.4-0.6=borderline, 0.85-1.0=clearly defective.""",
    ),

    # Variant C: calibrated scoring with concrete scale examples.
    "C_calibrated_scale": (
        _SYS + (
            " A clean uniform surface MUST score near 0.0; obvious damage MUST score near 1.0."
        ),
        """Rate the defect probability for this manufactured component image.

Scale reference:
  0.02 = smooth uniform surface, no issues whatsoever
  0.20 = minor natural variation, within normal tolerance
  0.50 = ambiguous — possible very subtle defect
  0.80 = likely defective — probable defect visible
  0.97 = clearly defective — obvious crack / hole / damage

Respond with ONLY valid JSON (no markdown, no extra text):
{"is_anomalous": <true/false>, "confidence": <float 0.0-1.0>, "defect_type": "<crack|scratch|contamination|hole|dent|discoloration|missing|none>", "description": "<surface observation>"}""",
    ),
}


# ── JSON parsing ──────────────────────────────────────────────────────────────
def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


def extract_confidence(raw: str) -> tuple[float | None, bool | None, str]:
    """Returns (confidence, is_anomalous, short_desc).  Robust to thinking blocks + text format."""
    t = _strip_thinking(raw).strip()

    for candidate in [
        t,
        re.sub(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", r"\1", t).strip(),
    ]:
        start = candidate.find("{")
        if start == -1:
            continue
        depth, in_str, prev, end = 0, False, "", -1
        for i, ch in enumerate(candidate[start:], start):
            if ch == '"' and prev != "\\":
                in_str = not in_str
            if not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            prev = ch
        if end == -1:
            continue
        try:
            d = json.loads(candidate[start : end + 1])
            conf = d.get("confidence")
            is_a = d.get("is_anomalous")
            if conf is not None:
                conf = max(0.0, min(1.0, float(conf)))
            if isinstance(is_a, str):
                is_a = is_a.lower() in ("true", "yes", "1")
            desc = d.get("description", d.get("defect_type", ""))
            return conf, is_a, str(desc)[:80]
        except Exception:
            continue

    # Regex fallback — handles both quoted JSON keys and plain text format
    # e.g. "confidence": 0.2  or  confidence: 0.2  or  Confidence: 0.2
    cm = re.search(r'"?[Cc]onfidence"?\s*:\s*([0-9]*\.?[0-9]+)', t)
    am = re.search(r'"?[Ii]s_anomalous"?\s*:\s*(true|false|yes|no|1|0)', t, re.I)
    if cm:
        conf = max(0.0, min(1.0, float(cm.group(1))))
        is_a = None
        if am:
            is_a = am.group(1).lower() in ("true", "yes", "1")
        return conf, is_a, "(text-format fallback)"
    return None, None, raw[:80]


# ── Direct mtmd inference (no create_chat_completion) ─────────────────────────
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

    Uses mtmd_helper_eval_chunk_single (not eval_chunks) to avoid the
    internal mtmd state corruption that causes eval_chunks to return -1
    on the 2nd+ calls within the same mtmd_ctx.

    Text chunks → llama.eval() (handles KV cache seq_rm safely)
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

            # Process each chunk one at a time
            # TEXT: llama.eval → properly updates n_tokens + input_ids
            # IMAGE: mtmd_helper_eval_chunk_single → writes vision KV embeddings
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
                        False,  # logits_last: last text chunk via llm.eval handles it
                        ctypes.byref(new_n_past),
                    )
                    if ret != 0:
                        raise RuntimeError(
                            f"mtmd_helper_eval_chunk_single error {ret} at chunk {ci}"
                        )
                    llm.n_tokens = new_n_past.value

        finally:
            mtmd_cpp.mtmd_input_chunks_free(chunks)
    finally:
        mtmd_cpp.mtmd_bitmap_free(bitmap)

    # ── Sampling loop ─────────────────────────────────────────────────────────
    # Image embeddings stay in KV cache because:
    #   eval([tok]) calls kv_cache_seq_rm(-1, n_tokens, ∞) which is always empty.
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
        tok = llm.sample()      # ridx=-1 → last computed logit
        if tok == eos:
            break
        tokens_out.append(tok)
        llm.eval([tok])         # advance KV by 1; kv_cache_seq_rm(n_tokens,∞) safe

    return llm.detokenize(tokens_out).decode("utf-8", errors="replace")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--threads",    type=int, default=8)
    args = ap.parse_args()

    # Validate paths & images
    for p, name in [(MODEL_PATH, "model GGUF"), (MMPROJ_PATH, "mmproj GGUF")]:
        if not p.exists():
            print(f"[ERROR] {name} not found: {p}", file=sys.stderr)
            sys.exit(1)

    missing = [str(p) for p, *_ in TEST_IMAGES if not p.exists()]
    if missing:
        print(f"[ERROR] missing test images:\n  " + "\n  ".join(missing), file=sys.stderr)
        sys.exit(1)

    # Load model — NO chat handler; we manage mtmd directly
    print(f"\n{'='*70}")
    print("  smoke_prompt_test.py  v3 — direct mtmd API")
    print(f"  Model  : {MODEL_PATH.name}")
    print(f"  Mmproj : {MMPROJ_PATH.name}")
    print(f"  Images : {len(TEST_IMAGES)}  |  Threads: {args.threads}  |  Max tokens: {args.max_tokens}")
    print(f"{'='*70}\n")

    import llama_cpp as _lc
    from llama_cpp import Llama, mtmd_cpp

    print("[load] Loading language model (no handler)…")
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
    mtmd_ctx = mtmd_cpp.mtmd_init_from_file(
        str(MMPROJ_PATH).encode(), llm.model, ctx_params
    )
    if mtmd_ctx is None:
        print("[ERROR] mtmd_init_from_file returned None — bad mmproj file?", file=sys.stderr)
        sys.exit(1)
    if not mtmd_cpp.mtmd_support_vision(mtmd_ctx):
        print("[ERROR] mtmd says vision not supported by this model.", file=sys.stderr)
        sys.exit(1)
    print(f"[load] mtmd ready in {time.time()-t0:.1f}s\n")

    # Results: prompt_name → list of (label, conf, is_anom, desc, raw_preview)
    results: dict[str, list[tuple]] = {k: [] for k in PROMPTS}

    for pname, (sys_msg, usr_msg) in PROMPTS.items():
        print(f"\n{'='*68}")
        print(f"  Prompt: {pname}")
        print(f"{'='*68}")

        for img_path, label, img_name in TEST_IMAGES:
            t0 = time.time()
            try:
                raw = infer_image(llm, mtmd_ctx, img_path, sys_msg, usr_msg,
                                  max_tokens=args.max_tokens)
                elapsed = time.time() - t0
                conf, is_anom, desc = extract_confidence(raw)

                gt_s   = "ANOM" if label else "GOOD"
                conf_s = f"{conf:.3f}" if conf is not None else " ERR "
                pred_s = ("ANOM" if is_anom else "GOOD") if is_anom is not None else "  ? "
                ok     = (is_anom == (label == 1)) if is_anom is not None else False
                mark   = "✓" if ok else "✗"

                print(f"  {mark} [{gt_s}→{pred_s}] {img_name:22s}  conf={conf_s}  {elapsed:.0f}s")
                print(f"    desc: {desc[:90]}")
                raw_preview = raw[:300].replace("\n", "↵")
                print(f"    raw : {raw_preview}")
                results[pname].append((label, conf, is_anom, desc))

            except Exception as exc:
                elapsed = time.time() - t0
                print(f"  ERROR {img_name}: {exc}  ({elapsed:.0f}s)")
                results[pname].append((label, None, None, str(exc)[:60]))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n\n{'='*72}")
    print("  PROMPT COMPARISON  (ideal: normal→conf≈0.0, anomalous→conf≈1.0)")
    print(f"{'='*72}")
    print(f"  {'Prompt':<22} {'Anom-avg':>9} {'Norm-avg':>9} {'Gap':>8} {'Acc':>6}")
    print(f"  {'─'*58}")

    best_prompt, best_gap = None, -999.0
    for pname, rows in results.items():
        anom_c = [c for lbl, c, _, __ in rows if lbl == 1 and c is not None]
        norm_c = [c for lbl, c, _, __ in rows if lbl == 0 and c is not None]
        anom_a = sum(anom_c) / len(anom_c) if anom_c else float("nan")
        norm_a = sum(norm_c) / len(norm_c) if norm_c else float("nan")
        gap    = anom_a - norm_a
        acc    = sum(1 for lbl, c, ia, _ in rows
                     if ia is not None and (ia == (lbl == 1))) / max(len(rows), 1)
        print(f"  {pname:<22} {anom_a:>9.3f} {norm_a:>9.3f} {gap:>8.3f} {acc:>6.2f}")
        if gap > best_gap:
            best_gap, best_prompt = gap, pname

    print(f"\n  Best prompt: {best_prompt}  (gap={best_gap:.3f})")
    print(f"  → Use this prompt in run_minicpm_local_sweep.py for the full sweep.")
    print(f"{'='*72}\n")

    mtmd_cpp.mtmd_free(mtmd_ctx)


if __name__ == "__main__":
    main()

// LlavaSupport.h — Bridging header for the llama.cpp mtmd (multimodal) C API.
//
// This file is the Objective-C Bridging Header for VLMAnomalyEdge.
// It exposes mtmd.h functions to Swift via the #if LLAVA_AVAILABLE guard in
// MiniCPMRunner.swift.
//
// HOW TO ENABLE:
//   1. Build a custom llama.xcframework with mtmd compiled in:
//        bash scripts/build-ios-multimodal.sh
//      This outputs Frameworks/build-apple-mtmd/llama.xcframework
//   2. Update project.yml to reference the new xcframework instead of
//      Frameworks/build-apple/llama.xcframework
//   3. Re-run: xcodegen generate
//   4. In Xcode → Build Settings → Swift Compiler – Custom Flags:
//      Active Compilation Conditions → add LLAVA_AVAILABLE
//
// Without step 4, the #if LLAVA_AVAILABLE blocks are excluded and the app
// compiles in text-only mode (no image encoding, model answers from prompt text
// alone — useful for smoke-testing the GGUF load / generation loop).

#pragma once
#include <stdint.h>
#include <stdbool.h>

// ── Opaque types (mtmd.h, b9305+) ────────────────────────────────────────────
struct llama_model;    // from llama.h, already in the llama module
struct llama_context;  // from llama.h

struct mtmd_context;
struct mtmd_bitmap;
struct mtmd_image_tokens;
struct mtmd_input_chunk;
struct mtmd_input_chunks;

struct mtmd_input_text {
    const char * text;
    bool add_special;
    bool parse_special;
};

struct mtmd_context_params {
    bool use_gpu;
    bool print_timings;
    int  n_threads;
    const char * media_marker;    // default: "<__media__>"
    int  flash_attn_type;         // enum llama_flash_attn_type
    bool warmup;
    int  image_min_tokens;
    int  image_max_tokens;
};

#ifdef __cplusplus
extern "C" {
#endif

// ── mtmd context ──────────────────────────────────────────────────────────────
struct mtmd_context_params mtmd_context_params_default(void);
const char *               mtmd_default_marker(void);

struct mtmd_context * mtmd_init_from_file(
    const char                      * mmproj_fname,
    const struct llama_model        * text_model,
    const struct mtmd_context_params  ctx_params
);
void mtmd_free(struct mtmd_context * ctx);

// ── bitmap (RGB image data) ───────────────────────────────────────────────────
// data must be nx * ny * 3 bytes in RGBRGB... format
struct mtmd_bitmap * mtmd_bitmap_init(uint32_t nx, uint32_t ny, const unsigned char * data);
void                 mtmd_bitmap_free(struct mtmd_bitmap * bitmap);

// ── tokenise prompt + images ──────────────────────────────────────────────────
// Prompt must contain mtmd_default_marker() once per bitmap.
// Returns 0 on success, 1 on marker/bitmap count mismatch, 2 on encode error.
struct mtmd_input_chunks * mtmd_input_chunks_init(void);
void                       mtmd_input_chunks_free(struct mtmd_input_chunks * chunks);
size_t                     mtmd_input_chunks_size(const struct mtmd_input_chunks * chunks);
const struct mtmd_input_chunk * mtmd_input_chunks_get(const struct mtmd_input_chunks * chunks, size_t idx);

int32_t mtmd_tokenize(
    struct mtmd_context       * ctx,
    struct mtmd_input_chunks  * output,
    const struct mtmd_input_text * text,
    const struct mtmd_bitmap ** bitmaps,
    size_t                      n_bitmaps
);

// ── chunk inspection ──────────────────────────────────────────────────────────
// chunk type: 0 = text, 1 = image, 2 = audio
int           mtmd_input_chunk_get_type(const struct mtmd_input_chunk * chunk);
const int32_t * mtmd_input_chunk_get_tokens_text(const struct mtmd_input_chunk * chunk, size_t * n_tokens_output);
size_t        mtmd_input_chunk_get_n_tokens(const struct mtmd_input_chunk * chunk);
int32_t       mtmd_input_chunk_get_n_pos(const struct mtmd_input_chunk * chunk);

// ── encode + get embeddings ───────────────────────────────────────────────────
int32_t mtmd_encode_chunk(struct mtmd_context * ctx, const struct mtmd_input_chunk * chunk);
float * mtmd_get_output_embd(struct mtmd_context * ctx);

bool mtmd_support_vision(const struct mtmd_context * ctx);

#ifdef __cplusplus
}
#endif

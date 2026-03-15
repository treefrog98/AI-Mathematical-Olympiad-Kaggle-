"""
AIMO3 Inference + Dev Loop
--------------------------
Run this on RunPod to benchmark models against reference.csv.

Usage:
    python inference.py                        # runs MODEL_KEY below
    python inference.py --model deepseek-r1-7b # override from CLI

Install:
    pip install vllm polars
"""

import re
import time
import argparse
import polars as pl
from vllm import LLM, SamplingParams

# ── 1. Model registry ──────────────────────────────────────────────────────────
# Add new models here as you experiment. Key = short name, value = HF model ID.
MODELS = {
    "deepseek-r1-7b":  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",   # 14GB VRAM fp16
    "qwen-math-7b":    "Qwen/Qwen2.5-Math-7B-Instruct",              # 14GB VRAM fp16
    "deepseek-r1-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",   # needs ~20GB (4bit) or 80GB GPU
}

# ── 2. Change this to switch models ───────────────────────────────────────────
MODEL_KEY = "deepseek-r1-7b"

# ── 3. Generation settings ─────────────────────────────────────────────────────
# Keeping these at the top makes them easy to tune without hunting through code.
# More tokens = more thinking time = better answers on hard problems.
# 4096 was too low — the model was getting cut off mid-reasoning, causing garbage answers.
MAX_NEW_TOKENS = 16384
TEMPERATURE    = 0.6     # 0.0 = greedy/deterministic, 1.0 = very random, 0.6 = slightly creative


# ── 4. Load model ──────────────────────────────────────────────────────────────
def load_model(model_key: str) -> LLM:
    """
    vLLM's LLM object replaces both the HuggingFace model AND tokenizer.
    It handles tokenization, device placement, and generation internally.

    Why vLLM instead of HuggingFace model.generate()?
    - PagedAttention: smarter GPU memory management for the KV cache
      (KV cache = intermediate attention computations stored as tokens generate)
    - Optimized CUDA kernels: lower-level GPU ops than HuggingFace's generic ones
    - Result: 3-5x faster generation, same output quality

    config.json and model.safetensors work exactly the same under the hood:
    - config.json = blueprint (architecture shape)
    - model.safetensors = actual trained weights
    vLLM still reads both, just much more efficiently than HuggingFace.

    max_model_len = max total tokens (input + output) the model will handle in one call.
    This is a memory budget — larger = more VRAM used for the KV cache.
    dtype="half" = fp16, uses half the memory of fp32 with negligible accuracy loss.
    """
    model_id = MODELS[model_key]
    print(f"\nLoading: {model_id}")

    if model_key == "deepseek-r1-32b":
        # 32B in fp16 = ~64GB, won't fit on a 3090 (24GB).
        # bitsandbytes 4-bit quantization shrinks it to ~16GB.
        # Weights stored as 4-bit integers, math ops still done in fp16 for accuracy.
        llm = LLM(
            model=model_id,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            dtype="half",
            max_model_len=32768,
        )
    else:
        llm = LLM(
            model=model_id,
            dtype="half",
            max_model_len=32768,
        )

    print("Model loaded!\n")
    return llm


# ── 5. Run inference on one problem ───────────────────────────────────────────
def run_inference(llm: LLM, problem: str) -> str:
    """
    vLLM's .chat() handles the chat template formatting automatically —
    same job that apply_chat_template did in HuggingFace, but built in.

    Every model has its own chat format. DeepSeek-R1 internally formats as:
    <|User|>What is 2+2?<|Assistant|><think>
    .chat() reads the model's tokenizer_config.json to apply the right format.

    SamplingParams controls how tokens are sampled during generation:
    - temperature: controls randomness of next-token selection
      0.0 = always pick the single most likely token (greedy, deterministic)
      1.0 = sample proportionally to model's full probability distribution
      0.6 = in between — some diversity but not chaotic
    - max_tokens: the model's "thinking budget" — how many tokens it can generate
      For reasoning models, this IS the compute budget. More tokens = more thinking steps.

    outputs[0].outputs[0].text returns ONLY the generated text (no input included).
    vLLM does this cleanly — no need to manually slice off the input like HuggingFace does.
    """
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    )

    outputs = llm.chat(
        messages=[{"role": "user", "content": problem}],
        sampling_params=sampling_params,
    )

    return outputs[0].outputs[0].text


# ── 6. Extract the final integer answer from raw output ───────────────────────
def extract_answer(raw_output: str) -> int | None:
    """
    The model outputs a wall of text (full reasoning trace). The competition
    needs a single integer. This function extracts it.

    Example raw output from DeepSeek-R1:
    <think>
    Let me work through this step by step...
    [hundreds of lines of reasoning with intermediate numbers]
    ...therefore the answer is 50.
    </think>
    The product of Alice and Bob's ages is **50**.

    We strip <think>...</think> FIRST because the reasoning block contains
    hundreds of intermediate numbers (wrong attempts, partial calculations).
    If we don't strip it, Pattern 3 grabs one of those instead of the final answer.

    Patterns tried in order of reliability:
    1. \boxed{1234} — LaTeX boxed notation, standard in math competitions
    2. "answer is 1234" / "answer: 1234" — plain English fallback
    3. Last integer in the output — last resort, often the final answer is last
    """
    output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

    # Pattern 1: \boxed{1234}
    boxed = re.findall(r"\\boxed\{(\d+)\}", output)
    if boxed:
        return int(boxed[-1])

    # Pattern 2: "answer is 1234" or "answer: 1234"
    answer_phrase = re.findall(r"answer(?:\s+is)?[:\s]+(\d+)", output, re.IGNORECASE)
    if answer_phrase:
        return int(answer_phrase[-1])

    # Pattern 3: last standalone integer in the output
    all_ints = re.findall(r"\b(\d{1,6})\b", output)
    if all_ints:
        return int(all_ints[-1])

    return None


# ── 7. Dev loop ────────────────────────────────────────────────────────────────
def run_dev_loop(llm: LLM, model_key: str, reference_path: str = "reference.csv"):
    ref = pl.read_csv(reference_path)
    correct = 0
    total = len(ref)
    results = []

    print(f"{'='*65}")
    print(f"Benchmarking {model_key} on {total} reference problems")
    print(f"max_new_tokens={MAX_NEW_TOKENS} | temperature={TEMPERATURE}")
    print(f"{'='*65}\n")

    for i, row in enumerate(ref.iter_slices(n_rows=1)):
        id_      = row["id"].item(0)
        problem  = row["problem"].item(0)
        true_ans = row["answer"].item(0)

        t0         = time.time()
        raw        = run_inference(llm, problem)
        elapsed    = time.time() - t0
        pred       = extract_answer(raw)
        is_correct = pred == true_ans
        correct   += is_correct

        status = "CORRECT" if is_correct else "WRONG"
        print(f"[{i+1:2d}/{total}] {status} | id={id_} | pred={pred} | true={true_ans} | {elapsed:.0f}s")

        # Save full raw_output so you can inspect reasoning traces after the run
        results.append({
            "id":          id_,
            "true_answer": true_ans,
            "predicted":   pred,
            "correct":     is_correct,
            "raw_output":  raw,
            "elapsed_sec": round(elapsed, 1),
        })

    print(f"\n{'='*65}")
    print(f"Score: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'='*65}\n")

    out_path = f"/workspace/results_{model_key}.csv"
    pl.DataFrame(results).write_csv(out_path)
    print(f"Full outputs saved to: {out_path}")


# ── 8. Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_KEY, choices=MODELS.keys())
    args = parser.parse_args()

    llm = load_model(args.model)
    run_dev_loop(llm, args.model)

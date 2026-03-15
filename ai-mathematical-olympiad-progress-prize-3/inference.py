"""
AIMO3 Inference + Dev Loop
--------------------------
Run this on RunPod to benchmark models against reference.csv.

Usage:
    python inference.py                         # runs MODEL_KEY below
    python inference.py --model deepseek-r1-32b # override from CLI

Install:
    pip install vllm polars bitsandbytes
"""

import re
import time
import argparse
from collections import Counter
from pathlib import Path

import polars as pl
from vllm import LLM, SamplingParams

# Always resolve paths relative to this script's directory, regardless of cwd.
SCRIPT_DIR = Path(__file__).parent


# ── 1. Model registry ──────────────────────────────────────────────────────────
MODELS = {
    "deepseek-r1-7b":  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",            # ~14GB VRAM fp16
    "deepseek-r1-32b": "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit",      # ~20GB VRAM 4bit
}

# A100 80GB has enough headroom for 32768 on both models.
# If you're on a smaller GPU (e.g. RTX Pro 4500 32GB), reduce deepseek-r1-32b to 28000.
MODEL_MAX_LEN = {
    "deepseek-r1-7b":  32768,
    "deepseek-r1-32b": 32768,
}

# ── 2. Change this to switch models ───────────────────────────────────────────
MODEL_KEY = "deepseek-r1-7b"

# ── 3. Generation settings ─────────────────────────────────────────────────────
MAX_NEW_TOKENS       = 16384  # output token budget per sample
TEMPERATURE          = 0.6    # 0.0 = greedy, 0.6 = slightly creative (good for math diversity)
BATCH_SIZE           = 4      # samples generated per round in adaptive voting
MAX_SAMPLES          = 32     # hard cap on samples per problem
CONFIDENCE_THRESHOLD = 0.75   # stop early when this fraction of samples agree


# ── 4. System prompt ───────────────────────────────────────────────────────────
# Design rationale:
# - No few-shot examples: DeepSeek-R1 already does CoT inside <think> tags.
#   Few-shot wastes 1000+ input tokens per sample and can anchor reasoning style.
# - State answer range explicitly: 8/10 reference problems ask for a remainder
#   (mod 10^5 or mod 99991), so the answer is always in [0, 99999].
# - Remind about remainders: the most common mistake is giving the raw value
#   instead of the modular result.
SYSTEM_PROMPT = """\
You are an expert competition mathematician. Solve the following problem from \
the AI Mathematical Olympiad (AIMO).

These problems span number theory, combinatorics, geometry, algebra, and \
functional equations. The answer is always a non-negative integer. When the \
problem asks for a remainder (e.g. "divided by 10^5" or "divided by 99991"), \
compute that remainder — do not give the raw value.

Work methodically: identify structure, consider modular arithmetic or \
generating functions where applicable, verify with small cases. Your final \
answer must be a single integer between 0 and 99999, written as \\boxed{N}.\
"""


# ── 5. Load model ──────────────────────────────────────────────────────────────
def load_model(model_key: str) -> LLM:
    """
    Load a vLLM engine for the given model.

    vLLM vs HuggingFace generate:
    - PagedAttention: KV cache blocks are allocated dynamically per token,
      not pre-allocated per sequence. This lets many sequences share GPU memory
      efficiently, which is critical for majority voting (many concurrent samples).
    - Prefix caching: when multiple samples share the same prompt, vLLM computes
      and caches the prompt KV cache once and reuses it for all N samples.
      For majority voting, this means you only pay for output tokens N times.
    """
    model_id = MODELS[model_key]
    max_model_len = MODEL_MAX_LEN[model_key]
    print(f"\nLoading: {model_id}")

    if model_key == "deepseek-r1-32b":
        # bitsandbytes 4-bit: weights stored as 4-bit integers, ops done in fp16.
        # Shrinks 32B from ~64GB fp16 → ~20GB in VRAM.
        llm = LLM(
            model=model_id,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            dtype="half",
            max_model_len=max_model_len,
        )
    else:
        llm = LLM(
            model=model_id,
            dtype="half",
            max_model_len=max_model_len,
        )

    print("Model loaded!\n")
    return llm


# ── 6. Build chat messages ─────────────────────────────────────────────────────
def build_messages(problem: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": problem},
    ]


# ── 7. Generate a batch of N samples for one problem ──────────────────────────
def generate_batch(llm: LLM, problem: str, n: int, max_new_tokens: int) -> list[str]:
    """
    Submit N samples simultaneously to vLLM in a single call.

    Passing [messages] * n lets vLLM schedule all N sequences together and
    apply prefix caching — the prompt KV cache is computed once and shared
    across all N samples. GPU stays fully utilised throughout generation.

    Contrast with: calling llm.chat() N times in a loop → GPU sits idle
    between calls, no prefix sharing, ~N× slower.
    """
    messages = build_messages(problem)
    sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=max_new_tokens)
    outputs = llm.chat(
        messages=[messages] * n,
        sampling_params=sampling_params,
    )
    return [o.outputs[0].text for o in outputs]


# ── 8. Extract integer answer from raw output ─────────────────────────────────
def extract_answer(raw_output: str) -> int | None:
    """
    Strip the <think> block first — it contains hundreds of intermediate
    numbers (wrong attempts, scratch work) that would confuse patterns 2 and 3.

    Patterns tried in order of reliability:
    1. \\boxed{1234}  — LaTeX notation, standard in math competitions
    2. "answer is 1234" / "answer: 1234"  — plain English fallback
    3. Last integer in the output  — last resort
    """
    output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

    boxed = re.findall(r"\\boxed\{(\d+)\}", output)
    if boxed:
        return int(boxed[-1])

    answer_phrase = re.findall(r"answer(?:\s+is)?[:\s]+(\d+)", output, re.IGNORECASE)
    if answer_phrase:
        return int(answer_phrase[-1])

    all_ints = re.findall(r"\b(\d{1,6})\b", output)
    if all_ints:
        return int(all_ints[-1])

    return None


# ── 9. Adaptive majority voting ───────────────────────────────────────────────
def adaptive_majority_vote(
    llm: LLM,
    problem: str,
    max_new_tokens: int,
    max_samples: int = MAX_SAMPLES,
    batch_size: int = BATCH_SIZE,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> tuple[int | None, int, Counter]:
    """
    Time-budgeted majority voting: allocate more samples to uncertain problems,
    fewer to problems where the model converges quickly.

    Algorithm:
    1. Generate batch_size samples.
    2. Compute confidence = count(mode answer) / total samples.
    3. If confidence >= threshold → stop early (model has converged).
    4. Else → generate another batch and repeat.
    5. Hard cap at max_samples.

    Why this works well for olympiad problems:
    - Easy problems: model gives the same answer 3/4 times → exits after 4 samples.
    - Hard problems: model disagrees with itself → uses up to 32 samples.
    - Hard problems also generate longer reasoning chains per sample, so the
      dynamic budget keeps total wall-clock time roughly proportional to difficulty
      rather than uniformly wasting compute on easy problems.

    Returns: (final_answer, n_samples_used, vote_counts)
    """
    answers: list[int] = []

    while len(answers) < max_samples:
        remaining = max_samples - len(answers)
        batch_raw = generate_batch(llm, problem, min(batch_size, remaining), max_new_tokens)
        parsed = [extract_answer(r) for r in batch_raw]
        answers.extend(a for a in parsed if a is not None)

        # Check for early exit after at least one full batch
        if len(answers) >= batch_size:
            counts = Counter(answers)
            mode_count = counts.most_common(1)[0][1]
            confidence = mode_count / len(answers)
            if confidence >= confidence_threshold:
                break

    counts = Counter(answers)
    final_answer = counts.most_common(1)[0][0] if counts else None
    return final_answer, len(answers), counts


# ── 10. Dev loop ───────────────────────────────────────────────────────────────
def run_dev_loop(llm: LLM, model_key: str, reference_path: str | None = None):
    if reference_path is None:
        reference_path = SCRIPT_DIR / "reference.csv"
    ref = pl.read_csv(reference_path)
    correct = 0
    total = len(ref)
    results = []

    # Leave 512 tokens of headroom for the input prompt.
    max_new_tokens = min(MAX_NEW_TOKENS, MODEL_MAX_LEN[model_key] - 512)

    print(f"{'='*70}")
    print(f"Benchmarking: {model_key}  |  problems: {total}")
    print(f"max_new_tokens={max_new_tokens}  |  max_samples={MAX_SAMPLES}  |  "
          f"confidence_threshold={CONFIDENCE_THRESHOLD}  |  batch_size={BATCH_SIZE}")
    print(f"{'='*70}\n")

    for i, row in enumerate(ref.iter_slices(n_rows=1)):
        id_      = row["id"].item(0)
        problem  = row["problem"].item(0)
        true_ans = row["answer"].item(0)

        t0 = time.time()
        pred, n_samples, vote_counts = adaptive_majority_vote(llm, problem, max_new_tokens)
        elapsed = time.time() - t0

        is_correct = pred == true_ans
        correct   += is_correct

        status = "CORRECT" if is_correct else "WRONG"
        print(
            f"[{i+1:2d}/{total}] {status} | id={id_} | pred={pred} | true={true_ans} | "
            f"samples={n_samples} | votes={dict(vote_counts)} | {elapsed:.0f}s"
        )

        results.append({
            "id":          id_,
            "true_answer": true_ans,
            "predicted":   pred,
            "correct":     is_correct,
            "n_samples":   n_samples,
            "vote_counts": str(dict(vote_counts)),
            "elapsed_sec": round(elapsed, 1),
        })

    total_samples = sum(r["n_samples"] for r in results)
    print(f"\n{'='*70}")
    print(f"Score: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"Total samples used: {total_samples} / {total * MAX_SAMPLES} max  "
          f"({100*total_samples/(total*MAX_SAMPLES):.0f}% of budget)")
    print(f"{'='*70}\n")

    out_path = f"/workspace/results_{model_key}.csv"
    pl.DataFrame(results).write_csv(out_path)
    print(f"Results saved to: {out_path}")


# ── 11. Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_KEY, choices=MODELS.keys())
    args = parser.parse_args()

    llm = load_model(args.model)
    run_dev_loop(llm, args.model)

"""
AIMO3 Inference + Dev Loop
--------------------------
Run this on RunPod to benchmark models against reference.csv.

Usage:
    python inference.py                        # runs MODEL_KEY below
    python inference.py --model deepseek-r1-7b # override from CLI
"""

import re
import time
import argparse
import polars as pl
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── 1. Model registry ──────────────────────────────────────────────────────────
# Add new models here as you experiment. Key = short name, value = HF model ID.
MODELS = {
    "deepseek-r1-7b":  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",   # 14GB VRAM fp16
    "qwen-math-7b":    "Qwen/Qwen2.5-Math-7B-Instruct",              # 14GB VRAM fp16
    "deepseek-r1-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",   # needs 4bit or 80GB GPU
}

# ── 2. Change this to switch models ───────────────────────────────────────────
MODEL_KEY = "deepseek-r1-7b"


# ── 3. Load model ──────────────────────────────────────────────────────────────
def load_model(model_key: str):
    model_id = MODELS[model_key]
    print(f"\nLoading: {model_id}")

    # For 32B, enable 4-bit quantization so it fits on smaller GPUs
    use_4bit = model_key == "deepseek-r1-32b"

    if use_4bit:
        """
        load_in_4bit - stores weights as 4-bit integers instead of 16-bit floats (4x smaller in memory)
        bnb_4bit_compute_dtype - store weights in 4-bit, actual math ops happen in fp16 for accuracy,
        so small storage, reasonable precision
        """
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        """
        1. downloads the weights from HF or loads from cache if already downloaded
        2. Builds the neural network architecture using config.json
        3. Fills that architecture with the weights, quantizing them on the fly if quantization_config
        is set.
        4. device_map="auto" - tells the `accelerate` library to figure out GPU placement automatically
        (on a single GPU it puts everything on cuda:0. On multiple GPUs it splits layers across them.)

        config.json - blueprint, weights are the nkowledge
        - config.json = architectural blueprint - describes the shape of the architecture
        - model.safetensors = the actual weights of the model

        e.g.
        {
            "num_hidden_layers": 28,       ← how many transformer layers deep
            "hidden_size": 3584,           ← how wide each layer is (neurons)
            "num_attention_heads": 28,     ← how many attention heads per layer
            "intermediate_size": 18944,    ← size of the feedforward network inside each layer (width of inner layer of FFN)
            "vocab_size": 152064,          ← how many tokens the model knows
        }
        Python first reads config.json and creates:
        → creates 28 empty transformer layers in memory
        → each layer has empty weight matrices of the right shape
        → model exists but is full of zeros/random numbers
        → useless at this point, just the skeleton

        Python reads model.safetensors
        → copies the actual trained numbers into those empty matrices
        → now the skeleton has a brain
        → model is ready to use
        """
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map="auto",
        )
    """
    Loads the tokenizer separately. The tokenizer is not part of the neural network — it's a lookup table that 
    converts text ↔ token IDs. It's lightweight (a few MB)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    """
    model.parameters() returns all weight tensors. next() grabs the first one. .device tells you where it lives — cuda:0 means GPU, 
    cpu means RAM. This is just a sanity check to confirm the model actually landed on the GPU.
    """
    print(f"Loaded on: {next(model.parameters()).device}\n")
    return model, tokenizer


# ── 4. Run inference on one problem ───────────────────────────────────────────
def run_inference(model, tokenizer, problem: str, max_new_tokens: int = 4096) -> str:
    """
    Chat models don't just take raw text. They expect a structured conversation format - a list of turns, each with a role (who is speaking)
    and content (what they said).
    """
    messages = [{"role": "user", "content": problem}]

    """
    Every model has its own special way of formatting a conversation into raw text before tokenizing.

    E.g. Deepseek-R1 handles internally like <|User|> What is 2+2?<|Assistant|><think>
    apply_chat_template handles that formatting automatically so you don't have to know the exact format for each model.

    - add_generation_prompt=True — adds the <｜Assistant｜> tag at the end, which tells the model "now it's your turn to respond"
    - return_tensors="pt" — return PyTorch tensors (numbers), not raw text. "pt" = PyTorch. Could also be "tf" for TensorFlow.
    - return_dict=True — return a dictionary {"input_ids": ..., "attention_mask": ...} instead of just a plain tensor
    - .to(model.device) — move those tensors onto the same device as the model (GPU). If the model is on cuda:0 and 
    your input is on CPU, it will crash.

    After this step, the problem text has been converted to a tensor of integers:
    "What is 2+2?" → [3923, 374, 220, 17, 10, 17, 30]

    Two keys:
    - input_ids — your text as numbers. Shape: [1, 23] meaning 1 sequence, 23 tokens long.
    - attention_mask — a tensor of 1s and 0s, same shape. 
    1 means "pay attention to this token", 0 means "ignore this (it's padding)". 
    For a single input it's all 1s. Only matters when you batch multiple sequences of different lengths together.
    """
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    """
    Remember how long the input is
    inputs["input_ids"] is a 2D tensor of shape [batch size, sequence length] <- -1 gets you the number of tokens in the input
    - we want to save this number when we decode the model's response (mark where the input ends so you know where the output begins)
    """
    input_len = inputs["input_ids"].shape[-1]

    """
    Generate the response

    - torch.no_grad() tells PyTorch not to track gradients - gradients we only need during training, we just waste memory and compute
    if it is on for inference
    - **inputs - unpacks the dictionary, so instead of passing one dict object, we pass in input_ids=..., attention_mask=... as separate
    keyword arguments
    - max_new_tokens - the model can generate at most 4k tokens. This should be high since math reasoning models can think a while before
    answering
    - temperature=0.6 - controls randomness (0.0 means pick single most likely next token, 1.0 sample proprotional to model's distribution)
    other values can just be creative
    - pad_token_id = tells the model what token means "I'm done" - without this some models throw a warning or generate forever

    model.generate returning inputs+outputs concatenated is a HF design decision
    - supports multiple use cases:
        - continued generation - sometimes you want to feed the output back in as a new input and keep generating (full sequence
        helps here)
        - encoder-decoder models behave differently, input/output separation is less clean there
        - beam search - internally tracks multiple candidate sequences, easier to return them all as full seqeuences
    """
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    """
    Decode only the new tokens

    = outputs[0] - grab the first (and only) item in the batch - now you have a 1D tensor of all the tokens
    - [input_len:] - slice off everything from input_len onwards (discards the input you sent and keeps only what the model generated)
    - tokenizer.decode(...) - converts token IDs back to human-readble text (reverse of what apply chat template did)
    - skip_special_tokens=True - removes special formatting tokens (we just want the plain text!)

    - outputs is still a 2D tensor [batch_size, sequence_length] - need to grab the first row and then slice the tokens
    """
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)


# ── 5. Extract the final integer answer from raw output ───────────────────────
def extract_answer(raw_output: str) -> int | None:
    """
    Math models typically end with \boxed{N} or a plain number.
    This tries a few patterns in order of reliability.

    When you call run_inference, you get back something like this:
    <think>
    Let me work through this step by step.

    Alice says if they each added their sweets to their age,
    her answer would be double Bob's. Let Alice have s_a sweets
    and age a, Bob have s_b sweets and age b.

    So s_a + a = 2(s_b + b)...

    [500 more lines of reasoning]

    ...therefore the product of their ages is 50.
    </think>

    The product of Alice and Bob's ages is **50**.
    The competition just needs: 50

    This function extracts that integer from the blob of text.
    """
    # Strip DeepSeek's <think>...</think> reasoning block first
    output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

    # Pattern 1: \boxed{1234} — most reliable, standard math format
    boxed = re.findall(r"\\boxed\{(\d+)\}", output)
    if boxed:
        return int(boxed[-1])  # take the last one

    # Pattern 2: "answer is 1234" or "answer: 1234"
    answer_phrase = re.findall(r"answer(?:\s+is)?[:\s]+(\d+)", output, re.IGNORECASE)
    if answer_phrase:
        return int(answer_phrase[-1])

    # Pattern 3: last standalone integer in the output
    all_ints = re.findall(r"\b(\d{1,6})\b", output)
    if all_ints:
        return int(all_ints[-1])

    return None  # failed to extract


# ── 6. Dev loop — runs all reference problems and scores the model ─────────────
def run_dev_loop(model, tokenizer, reference_path: str = "reference.csv"):
    ref = pl.read_csv(reference_path)
    correct = 0
    total = len(ref)
    results = []

    print(f"{'='*65}")
    print(f"Benchmarking on {total} reference problems")
    print(f"{'='*65}\n")

    for i, row in enumerate(ref.iter_slices(n_rows=1)):
        id_        = row["id"].item(0)
        problem    = row["problem"].item(0)
        true_ans   = row["answer"].item(0)

        t0         = time.time()
        raw        = run_inference(model, tokenizer, problem)
        elapsed    = time.time() - t0
        pred       = extract_answer(raw)
        is_correct = pred == true_ans
        correct   += is_correct

        status = "CORRECT" if is_correct else "WRONG"
        print(f"[{i+1:2d}/{total}] {status} | id={id_} | pred={pred} | true={true_ans} | {elapsed:.0f}s")

        # Save full output so you can inspect reasoning traces later
        results.append({
            "id": id_,
            "true_answer": true_ans,
            "predicted":   pred,
            "correct":     is_correct,
            "raw_output":  raw,
            "elapsed_sec": round(elapsed, 1),
        })

    print(f"\n{'='*65}")
    print(f"Score: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'='*65}\n")

    # Save results so you can inspect failures later
    pl.DataFrame(results).write_csv(f"results_{MODEL_KEY}.csv")
    print(f"Full outputs saved to: results_{MODEL_KEY}.csv")


# ── 7. Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_KEY, choices=MODELS.keys())
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    run_dev_loop(model, tokenizer)

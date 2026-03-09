import polars as pl

"""
Model Shortlist
1. deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

Distillation works like this:
- Train a huge powerful teacher model (DeepSeek-R1).
- Generate lots of reasoning examples from it.
- Train a smaller model to imitate the teacher.

2. Qwen/Qwen2.5-Math-7B-Instruct
- https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct

3. deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
- https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

Critical Dataset for SFT
https://huggingface.co/datasets/nvidia/OpenMathReasoning (used for AIMO-2 submission)
"""

# TODO: load your model here
def load_model():
    return None


# TODO: run inference with your model
def predict(model, problem: str) -> int:
    return 0


def run_dev_loop(reference_path: str = "reference.csv"):
    ref = pl.read_csv(reference_path)
    model = load_model()

    correct = 0
    total = len(ref)

    print(f"\n{'='*60}")
    print(f"Running on {total} reference problems")
    print(f"{'='*60}\n")

    for i, row in enumerate(ref.iter_slices(n_rows=1)):
        id_ = row["id"].item(0)
        problem_text = row["problem"].item(0)
        true_answer = row["answer"].item(0)

        pred = predict(model, problem_text)
        is_correct = pred == true_answer
        correct += is_correct

        status = "CORRECT" if is_correct else "WRONG"
        print(f"[{i+1:2d}/{total}] {status} | id={id_} | pred={pred} | true={true_answer}")

    print(f"\n{'='*60}")
    print(f"Score: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_dev_loop()

from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
from tqdm import tqdm
import os

# Load test dataset
test_ds = load_dataset("openai/gsm8k", "main", split="test")

# Load fine-tuned model
model_path = "./llama-star-finetuned/final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

print(f"Loaded model from: {model_path}")
print(f"Model device: {next(model.parameters()).device}")
print(f"Test samples: {len(test_ds)}")
print("=" * 50)

def extract_final_answer(text: str):
    """Extract the final numerical answer from the model output"""
    if not text:
        return None
    # Look for '#### <number>'
    m = re.search(r"####\s*([\-–—]?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if m:
        # Remove commas from numbers
        return m.group(1).replace(",", "").strip()
    # Fallback: last number in output
    nums = re.findall(r"([\-–—]?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if nums:
        return nums[-1].replace(",", "").strip()
    return None

@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens=512):
    """Generate completion from the model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    out = tokenizer.decode(gen[0], skip_special_tokens=True)
    # Remove the input prompt from output
    if out.startswith(prompt):
        out = out[len(prompt):]
    return out.strip()

# Run evaluation
questions, golds, pred_answers, pred_rationales = [], [], [], []

print("\nStarting evaluation...")
for ex in tqdm(test_ds, desc="Evaluating"):
    q = ex["question"].strip()
    gold = ex["answer"].split("####")[-1].strip().replace(",", "")
    
    # Format matches training: "Question: <question>\nAnswer: "
    prompt = f"Question: {q}\nAnswer:"
    
    # Generate completion
    completion = generate(model, tokenizer, prompt)
    
    # Extract predicted answer
    pred_num = extract_final_answer(completion) or ""
    
    questions.append(q)
    golds.append(gold)
    pred_answers.append(pred_num)
    pred_rationales.append(completion)

# Compute accuracy
correct = [int(p.strip() == g.strip()) for p, g in zip(pred_answers, golds)]
acc = sum(correct) / len(correct)

print("\n" + "=" * 50)
print(f"RESULTS")
print("=" * 50)
print(f"Total test samples: {len(correct)}")
print(f"Correct predictions: {sum(correct)}")
print(f"Exact Match Accuracy: {acc*100:.2f}%")
print("=" * 50)

# Save results
df = pd.DataFrame({
    "question": questions,
    "gold": golds,
    "pred_answer": pred_answers,
    "pred_rationale": pred_rationales,
    "correct": correct
})
output_file = "star_benchmark_results.csv"
df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# Show some examples
print("\n" + "=" * 50)
print("Sample Predictions:")
print("=" * 50)
# Show correct and incorrect examples
correct_examples = df[df['correct'] == 1].head(2)
incorrect_examples = df[df['correct'] == 0].head(2)

print("\n✓ CORRECT PREDICTIONS:")
for i, (idx, row) in enumerate(correct_examples.iterrows()):
    print(f"\nExample {i+1}:")
    print(f"Question: {row['question'][:150]}...")
    print(f"Gold Answer: {row['gold']}")
    print(f"Predicted Answer: {row['pred_answer']}")
    print(f"Rationale (first 200 chars): {row['pred_rationale'][:200]}...")
    print("-" * 50)

print("\n✗ INCORRECT PREDICTIONS:")
for i, (idx, row) in enumerate(incorrect_examples.iterrows()):
    print(f"\nExample {i+1}:")
    print(f"Question: {row['question'][:150]}...")
    print(f"Gold Answer: {row['gold']}")
    print(f"Predicted Answer: {row['pred_answer']}")
    print(f"Rationale (first 200 chars): {row['pred_rationale'][:200]}...")
    print("-" * 50)

print("\n" + "=" * 50)
print("Evaluation complete!")
print("=" * 50)
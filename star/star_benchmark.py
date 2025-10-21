from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
from tqdm import tqdm
import os

# Silence tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    attn_implementation="sdpa",
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
    nums = re.findall(r"([\-–—]?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if nums:
        return nums[-1].replace(",", "").strip()
    return None

@torch.inference_mode()
def generate_batch(model, tokenizer, prompts, max_new_tokens=512):
    """Generate completions for multiple prompts at once"""
    # Tokenize all prompts with padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,  
    ).to(model.device)
    
    # Generate for the entire batch
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Decode all outputs
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Remove the input prompts from outputs
    completions = []
    for prompt, output in zip(prompts, decoded_outputs):
        if output.startswith(prompt):
            completion = output[len(prompt):].strip()
        else:
            completion = output.strip()
        completions.append(completion)
    
    return completions

# Run evaluation with batching
questions, golds, pred_answers, pred_rationales = [], [], [], []

batch_size = 16

print("\nStarting batched evaluation...")
print(f"Batch size: {batch_size}")

# Process in batches
num_batches = (len(test_ds) + batch_size - 1) // batch_size

for i in tqdm(range(0, len(test_ds), batch_size), total=num_batches, desc="Evaluating"):
    # Get batch - FIX: HuggingFace dataset slicing returns dict of lists
    batch_end = min(i + batch_size, len(test_ds))
    batch = test_ds[i:batch_end]
    
    # Extract data from the batch dictionary
    batch_questions = [q.strip() for q in batch["question"]]
    batch_golds = [ans.split("####")[-1].strip().replace(",", "") for ans in batch["answer"]]
    batch_prompts = [f"Question: {q}\nAnswer:" for q in batch_questions]
    
    # Generate completions for entire batch
    completions = generate_batch(model, tokenizer, batch_prompts)
    
    # Process results
    for q, gold, completion in zip(batch_questions, batch_golds, completions):
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

# Print performance info
if torch.cuda.is_available():
    print("\nGPU Memory Usage:")
    print(f"Peak memory allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
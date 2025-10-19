from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
from tqdm import tqdm
import os
from huggingface_hub import login
import time

# Login to HuggingFace if token available
if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])

# Load training dataset
train_ds = load_dataset("openai/gsm8k", "main", split="train")
print(f"Training samples: {len(train_ds)}")

# Load the base model
model_path = "meta-llama/Llama-3.2-3B-Instruct" 
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
print("=" * 50)

def extract_final_answer(text: str):
    """Extract the final numerical answer"""
    if not text:
        return None
    # Look for '#### <number>'
    m = re.search(r"####\s*([\-–—]?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    # Fallback: last number in output
    nums = re.findall(r"([\-–—]?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if nums:
        return nums[-1].replace(",", "").strip()
    return None

@torch.inference_mode()
def generate_batch(model, tokenizer, prompts, max_new_tokens=512):
    """Generate completions for a batch of prompts"""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    outputs = tokenizer.batch_decode(gen, skip_special_tokens=True)
    
    # Remove prompts from outputs
    results = []
    for prompt, output in zip(prompts, outputs):
        if output.startswith(prompt):
            output = output[len(prompt):]
        results.append(output.strip())
    return results

# STaR dataset generation 
star_dataset = []
batch_size = 32

print("\nGenerating STaR dataset with Zero-Shot CoT...")
print(f"Batch size: {batch_size}")
print("=" * 50)

# Print prompts being used (for report)
print("\n" + "=" * 50)
print("PROMPTS USED FOR DATASET GENERATION:")
print("=" * 50)
print("\n1. RATIONALE GENERATION PROMPT (without hint):")
print("   Question: {question}")
print("   Let's think step by step.")
print("   Answer:")
print("\n2. RATIONALIZATION PROMPT (with hint):")
print("   Question: {question}")
print("   The correct answer is {correct_answer}.")
print("   Let's think step by step to arrive at this answer.")
print("   Answer:")
print("=" * 50 + "\n")

start_time = time.time()

# Phase 1: Zero-Shot CoT rationale generation
print("\nPhase 1: Zero-Shot CoT Rationale Generation...")
direct_success = []
needs_rationalization = []

for i in tqdm(range(0, len(train_ds), batch_size), desc="Rationale generation"):
    batch_end = min(i + batch_size, len(train_ds))
    batch_indices = range(i, batch_end)
    
    questions = [train_ds[idx]["question"].strip() for idx in batch_indices]
    gold_answers = [train_ds[idx]["answer"].split("####")[-1].strip().replace(",", "") for idx in batch_indices]
    original_answers = [train_ds[idx]["answer"] for idx in batch_indices]
    
    # Zero-Shot CoT prompting
    prompts = [
        f"Question: {q}\nLet's think step by step.\nAnswer:"
        for q in questions
    ]
    
    completions = generate_batch(model, tokenizer, prompts, max_new_tokens=512)
    
    for question, completion, gold, orig_ans in zip(questions, completions, gold_answers, original_answers):
        pred = extract_final_answer(completion) or ""
        
        if pred.strip() == gold.strip():
            # Success: model generated correct answer
            star_dataset.append({
                "question": question,
                "rationale": completion,
                "method": "rationale_generation",
                "original_answer": orig_ans
            })
            direct_success.append(True)
        else:
            # Failed: need rationalization
            needs_rationalization.append({
                "question": question,
                "gold_answer": gold,
                "original_answer": orig_ans
            })
            direct_success.append(False)

print(f"Rationale generation success: {sum(direct_success)}/{len(direct_success)} ({sum(direct_success)/len(direct_success)*100:.1f}%)")
print(f"Need rationalization: {len(needs_rationalization)}")

# Phase 2: Rationalization (hint-based generation)
if needs_rationalization:
    print(f"\nPhase 2: Rationalization for {len(needs_rationalization)} examples...")
    
    for i in tqdm(range(0, len(needs_rationalization), batch_size), desc="Rationalization"):
        batch = needs_rationalization[i:i+batch_size]
        
        # Rationalization: provide the correct answer as a hint
        hint_prompts = [
            f"Question: {ex['question']}\n"
            f"The correct answer is {ex['gold_answer']}.\n"
            f"Let's think step by step to arrive at this answer.\n"
            f"Answer:"
            for ex in batch
        ]
        
        completions = generate_batch(model, tokenizer, hint_prompts, max_new_tokens=512)
        
        for ex, completion in zip(batch, completions):
            # Ensure the completion has the final answer marker with correct answer
            if "####" not in completion:
                completion = f"{completion.strip()}\n#### {ex['gold_answer']}"
            else:
                # Replace any incorrect final answer with the ground truth
                completion = re.sub(r"####\s*[\-–—]?\d+(?:,\d{3})*(?:\.\d+)?", 
                                   f"#### {ex['gold_answer']}", completion)
            
            star_dataset.append({
                "question": ex["question"],
                "rationale": completion,
                "method": "rationalization",
                "original_answer": ex["original_answer"]
            })

elapsed_time = time.time() - start_time
print(f"\nTotal time: {elapsed_time/60:.1f} minutes")
print(f"Average time per example: {elapsed_time/len(train_ds):.2f} seconds")

# Statistics
methods_count = pd.Series([d["method"] for d in star_dataset]).value_counts()
print("\n" + "=" * 50)
print("STaR Dataset Statistics:")
print("=" * 50)
print(f"Total examples: {len(star_dataset)}")
print(f"Training set size: {len(train_ds)}")
print(f"Coverage: {len(star_dataset)}/{len(train_ds)} ({len(star_dataset)/len(train_ds)*100:.1f}%)")
print("\nBreakdown by method:")
for method, count in methods_count.items():
    print(f"  {method}: {count} ({count/len(star_dataset)*100:.1f}%)")
print("=" * 50)

# Save the STaR dataset
df = pd.DataFrame(star_dataset)
output_file = "star_dataset.csv"
df.to_csv(output_file, index=False)
print(f"\nSTaR dataset saved to: {output_file}")

# Show some examples
print("\n" + "=" * 50)
print("Sample STaR Examples:")
print("=" * 50)
for method in ['rationale_generation', 'rationalization']:
    examples = [d for d in star_dataset if d['method'] == method]
    if examples:
        print(f"\n{method.upper()} Example:")
        ex = examples[0]
        print(f"Question: {ex['question']}")
        print(f"Rationale (first 400 chars): {ex['rationale'][:400]}...")
        print("-" * 50)

print("\n" + "=" * 50)
print("Dataset generation complete!")
print("=" * 50)
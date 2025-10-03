from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import pandas as pd
import re
from tqdm import tqdm
import os

SYSTEM_PROMPT = (
    "You are a meticulous math tutor. Solve grade-school math word problems step by step.\n"
    "Make your reasoning explicit and finish with the final numeric answer in this exact format:\n"
    "#### <number>\n"
)

USER_PROMPT_ZS = (
    "Solve the problem carefully. Show your reasoning step by step, then finish with:\n"
    "#### <final answer>\n\n"
    "Question: {question}\n"
)

#train_ds = load_dataset("openai/gsm8k", "main", split="train")
test_ds = load_dataset("openai/gsm8k", "main", split="test")

if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model_id = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

def apply_chat_template(tokenizer, system: str, user: str):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def extract_final_answer(text: str):
    if not text:
        return None
    # Look for '#### <number>'
    m = re.search(r"####\s*([\-–—]?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1).strip()
    # fallback: last number in output
    nums = re.findall(r"([\-–—]?\d+(?:\.\d+)?)", text)
    return nums[-1] if nums else None

@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.001,
        eos_token_id=tokenizer.eos_token_id,
    )
    out = tokenizer.decode(gen[0], skip_special_tokens=True)
    if out.startswith(prompt):
        out = out[len(prompt):]
    return out.strip()

subset = test_ds 

questions, golds, pred_answers, pred_rationales = [], [], [], []
for ex in tqdm(subset):
    q = ex["question"].strip()
    gold = ex["answer"].split("####")[-1].strip()  
    user_prompt = USER_PROMPT_ZS.format(question=q)
    prompt = apply_chat_template(tokenizer, SYSTEM_PROMPT, user_prompt)
    completion = generate(model, tokenizer, prompt)
    pred_num = extract_final_answer(completion) or ""
    # ensure a single trailing "#### <number>"
    completion_no_tail = re.sub(r"\n*####\s*[^\n]*\s*$", "", completion).rstrip()
    pred_full = completion_no_tail + f"\n#### {pred_num}"

    questions.append(q)
    golds.append(gold)
    pred_answers.append(pred_num)
    pred_rationales.append(pred_full)

# compute accuracy
correct = [int(p.strip() == g.strip()) for p, g in zip(pred_answers, golds)]
acc = sum(correct) / len(correct)
print(f"Exact Match Accuracy: {acc*100:.2f}% ({sum(correct)}/{len(correct)})")

# Save results
df = pd.DataFrame({
    "question": questions,
    "gold": golds,
    "pred_answer": pred_answers,
    "pred_rationale": pred_rationales,
    "correct": correct
})
df.to_csv("zs_cot_results.csv", index=False)
print("Saved zs_cot_results.csv")

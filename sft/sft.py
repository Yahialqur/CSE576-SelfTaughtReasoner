from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from huggingface_hub import login
import os

# Check GPU availability
print("=" * 50)
print("GPU Configuration:")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Current GPU memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
else:
    raise RuntimeError("No GPU detected! Training will be very slow on CPU.")
print("=" * 50)

if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])

# Load dataset
train_ds = load_dataset("openai/gsm8k", "main", split="train")
eval_ds = load_dataset("openai/gsm8k", "main", split="test[:500]")
print(f"Training samples: {len(train_ds)}")
print(f"Validation samples: {len(eval_ds)}")

# Model configuration
model_id = "meta-llama/Llama-3.2-3B-Instruct"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Verify model is on GPU
print("\nModel Device Placement:")
print(f"Model device: {next(model.parameters()).device}")
print(f"GPU memory after model load: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print("=" * 50)

def format_example(example):
    """Simple question -> answer format"""
    question = example["question"].strip()
    answer = example["answer"].strip()
    
    # Simple concatenation: question + answer
    text = f"Question: {question}\nAnswer: {answer}{tokenizer.eos_token}"
    return {"text": text}

# Preprocess datasets
train_dataset = train_ds.map(format_example, remove_columns=train_ds.column_names)
eval_dataset = eval_ds.map(format_example, remove_columns=eval_ds.column_names)

def tokenize_function(examples):
    """Tokenize the text"""
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding=False,  
    )
    result["labels"] = result["input_ids"].copy()
    return result

# Tokenize datasets
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing training dataset"
)

tokenized_eval = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing evaluation dataset"
)

from transformers import default_data_collator
from dataclasses import dataclass
from typing import Any, Dict, List
import torch

@dataclass
class DataCollatorForCompletionOnlyLM:
    """Custom data collator that properly handles padding for both input_ids and labels"""
    tokenizer: Any
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Get max length in this batch
        max_length = max(len(f["input_ids"]) for f in features)
        
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        for feature in features:
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            
            # Calculate padding length
            padding_length = max_length - len(input_ids)
            
            # Pad input_ids and attention_mask
            padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            
            # Pad labels with -100 (ignore index)
            padded_labels = labels + [-100] * padding_length
            
            batch["input_ids"].append(padded_input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(padded_labels)
        
        # Convert to tensors
        batch = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
        
        return batch

# Data collator with custom padding
data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./llama-gsm8k-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    bf16=True,
    dataloader_num_workers=0, 
    optim="adamw_torch_fused",
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    weight_decay=0.01,
    dataloader_pin_memory=True,  
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

# Start training
print("\nStarting fine-tuning...")
print(f"Total parameters: {model.num_parameters():,}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Training device: {training_args.device}")
print("=" * 50)

trainer.train()

# Print final GPU memory usage
print("\nFinal GPU Memory Usage:")
print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
print(f"Max GPU memory allocated during training: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")

# Save the fine-tuned model
print("\nSaving model...")
trainer.save_model("./llama-gsm8k-finetuned/final")
tokenizer.save_pretrained("./llama-gsm8k-finetuned/final")

print("\nTraining complete! Model saved to ./llama-gsm8k-finetuned/final")
print("\nTo use the fine-tuned model:")
print("model = AutoModelForCausalLM.from_pretrained('./llama-gsm8k-finetuned/final')")
print("tokenizer = AutoTokenizer.from_pretrained('./llama-gsm8k-finetuned/final')")
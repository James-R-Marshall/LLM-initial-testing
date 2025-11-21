# ...existing code...
import os
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig

token = os.environ.get("HF_TOKEN")  # set HF_TOKEN in your environment if needed

max_seq_length = 2048
dtype = None

# Load model + tokenizer (4-bit to save VRAM)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,
)

# Load dataset (may require HF_TOKEN if private)
dataset = load_dataset("kjj0/4chanpol", token=token)
print("Loaded dataset:", dataset)

# Simple chat template compatible with TinyLlama chat style
tokenizer.chat_template = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% if message.role == 'system' %}<<SYS>>\n{% elif message.role == 'user' %}[INST] {% elif message.role == 'assistant' %} [/INST]\n{% endif %}"
    "{% if message.role == 'system' %}{{ message.content + '\\n' }}{% else %}{{ message.content }}{% endif %}"
    "{% endfor %}"
)

# Format function: try standard keys, fallback to raw text fields
def format_example(example):
    instr = example.get("instruction") or example.get("prompt") or ""
    inp = example.get("input") or ""
    out = example.get("output") or example.get("response") or example.get("text") or ""
    messages = [
        {"role": "system", "content": instr},
        {"role": "user", "content": inp},
        {"role": "assistant", "content": out},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

# Map dataset (adjust num_proc to your machine)
dataset = dataset.map(format_example, num_proc=4)
print("Example formatted text:", dataset["train"][0]["text"])

# Apply LoRA/PEFT
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
)

# Trainer config (tiny run example)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

# Train only on responses (keeps instruction formatting intact)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# Train and save
trainer_stats = trainer.train()
output_dir = "fine_tuned_tinyllama"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Saved fine-tuned model to: {output_dir}")
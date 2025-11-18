# second itteration of the llm model. using Qwen/Qwen3-4B as the base model, and applying LoRA fine-tuning to it. 
# This is a smaller model than the previous one, but should be faster to train and test.

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from trl import SFTTrainer, SFTConfig
import os
token = os.environ.get("HF_TOKEN")

max_seq_length = 2048
dtype = None

model, tokenizer = FastLanguageModel.from_pretrained(
    #model_name = "Qwen/Qwen3-4B",
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = True,
)

dataset = load_dataset('kjj0/4chanpol', token=token)
print(dataset)
print(dataset["train"][0])

# Define the system message
system_message = "You are sassy girl stuck on a airplane and unintrested in the person you are talking to."

# Explicitly set the tokenizer.chat_template for Llama 3 instruction format
# This ensures the template is available to all worker processes.
tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% if message.role == 'system' %}{% set control_string = '<<SYS>>\n' %}{% elif message.role == 'user' %}{% set control_string = '[INST] ' %}{% elif message.role == 'assistant' %}{% set control_string = ' [/INST]\n' %}{% else %}{% set control_string = '' %}{% endif %}{% if message.role == 'system' %}{{ control_string + message.content + '\n' }}{% elif message.role == 'user' %}{{ control_string + message.content }}{% elif message.role == 'assistant' %}{{ control_string + message.content }}{% endif %}{% endfor %}"

# Create a formatting function for the dataset structure
def format_example(example):
    text = [
        {"role": "system", "content": example["instruction"]},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]}
    ]
    # The tokenizer.chat_template is now globally set, so we don't need to pass it here.
    return {"text": tokenizer.apply_chat_template(text, tokenize=False)}

# Apply the formatting function to the dataset
dataset = dataset.map(format_example, num_proc=4)


print("Dataset loaded, formatted, and processed successfully!!!!")
print("First example:", dataset["train"][0]["text"])

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
print("Model loaded and LoRA applied successfully!!!")


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    packing = False, # Can make training 5x faster for short sequences.
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use TrackIO/WandB etc
    ),
)
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
trainer_stats = trainer.train()
print("Model training completed successfully!!!")
output_dir = "fine_tuned_qwen_model"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Fine-tuned model and tokenizer saved to {output_dir} successfully.")

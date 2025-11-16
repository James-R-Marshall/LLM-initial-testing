# to run this script you need to:
# pip install unsloth transformers accelerate peft trl datasets
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import os
token = os.environ.get("HF_TOKEN")
max_seq_length = 2048
dtype = None

# Load 4bit quantized Llama 3.1 8B model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Meta-Llama-3-8B-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = True,
)
print("Unsloth, transformers, accelerate, and peft installed successfully and Llama 3.1 8B model and tokenizer loaded successfully.")


# Load a widely available instruction-following dataset
#dataset = load_dataset('CausalLM/Refined-Anime-Text')
#dataset = load_dataset('lesserfield/4chan-datasets')
dataset = load_dataset('kjj0/4chanpol', token=token)

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
    return {"text": tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=False)}

# Apply the formatting function to the dataset
dataset = dataset.map(format_example, num_proc=4)

print("Dataset loaded, formatted, and processed successfully.")
print(dataset["train"][0]["text"])

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout = 0.1,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)
print(model.print_trainable_parameters())

# Define TrainingArguments
training_arguments = TrainingArguments(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    max_steps = 60,
    learning_rate = 2e-4,
    fp16 = True,
    logging_steps = 1,
    output_dir = "outputs",
    optim = "paged_adamw_8bit",
    seed = 3407,
)

# Initialize SFTTrainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = training_arguments,
)

print("SFTTrainer configured successfully.")

trainer.train()
print("Model training initiated successfully.")
output_dir = "fine_tuned_llama_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Fine-tuned model and tokenizer saved to {output_dir} successfully.")
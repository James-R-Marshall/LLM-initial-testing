import argparse
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# /home/james/LLM-initial-testing/test.py
# Simple loader for a model produced by a "llama-Lora.py" training script.
# Tries to load a full pretrained model directory first; if that fails and
# the directory looks like a LoRA adapter, it will ask for a base model
# and load the LoRA adapters on top using peft.
#
# Usage: python test.py
# You will be prompted for paths and for a generation prompt.


try:
    _HAS_PEPT = True
except Exception:
    _HAS_PEPT = False

def load_model_candidate(path, device, dtype):
    # Try loading as a full model first
    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True), \
               AutoModelForCausalLM.from_pretrained(path, device_map="auto" if device == "cuda" else None, torch_dtype=dtype, trust_remote_code=True)
    except Exception as e:
        raise e

def load_with_lora(adapter_path, base_path, device, dtype):
    if not _HAS_PEPT:
        raise RuntimeError("peft is required to load LoRA adapters. Install with: pip install peft")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(base_path, device_map="auto" if device == "cuda" else None, torch_dtype=dtype, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter_path, device_map="auto" if device == "cuda" else None)
    return tokenizer, model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Path to the model or LoRA adapter directory (default: ./lora_model)", default="./lora_model")
    parser.add_argument("--base", "-b", help="(Optional) base model path if --path is LoRA adapters")
    parser.add_argument("--device", help="cpu or cuda or auto (default auto)", default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_path = args.path
    tokenizer = model = None

    # Attempt direct load
    try:
        tokenizer, model = load_model_candidate(model_path, device, dtype, device_map="auto")
    except Exception:
        # If direct load failed, try to treat path as LoRA adapters
        if args.base:
            try:
                tokenizer, model = load_with_lora(model_path, args.base, device, dtype)
            except Exception as e:
                print("Failed to load LoRA-on-base:", e, file=sys.stderr)
                sys.exit(1)
        else:
            print("Direct load failed and no --base provided. Provide the base model path to load LoRA adapters.", file=sys.stderr)
            sys.exit(1)

    model.eval()
    # move tokenizer/model to device if needed (peft/transformers should handle device_map="auto")
    print("Model loaded. Device:", device)
    try:
        while True:
            prompt = input("\nEnter prompt (empty to quit):\n> ")
            if not prompt:
                break
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.7, top_p=0.95)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            print("\n=== OUTPUT ===\n")
            print(text)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")

if __name__ == "__main__":
    main()
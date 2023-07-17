import argparse
import os

os.environ['HF_DATASETS_OFFLINE'] = '1'  # ask datasets library not to make arbitrary web requests
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # ask transformer library not to make arbitrary web requests
os.environ['BITSANDBYTES_NOWELCOME'] = '1'  # disable welcome message that bitsandbytes prints, it's unnecessary noise

import torch
import peft
import transformers

def merge_lora_back_into_base_model(lora_path, base_model_path, output_path):
    peft_config = peft.PeftConfig.from_pretrained(lora_path)
    if peft_config.task_type == peft.TaskType.SEQ_CLS:
        base_model = transformers.AutoModelForSequenceClassification.from_pretrained(base_model_path, num_labels=1, use_safetensors=True)
    elif peft.TaskType.CAUSAL_LM:
        base_model = transformers.AutoModelForCausalLM.from_pretrained(base_model_path, return_dict=True, use_safetensors=True)
    peft_base_model = peft.PeftModel.from_pretrained(base_model, lora_path)
    peft_base_model.eval()  # switch the model over into inference mode, disabling training-specific functionality such as dropout layers
    merged_model = peft_base_model.merge_and_unload()
    merged_model.save_pretrained(output_path)

    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lora_path", type=str, help=f"Path to the directory containing adapter_model.safetensors")
    parser.add_argument("base_model_path", type=str, help=f"Path to the directory containing config.json")
    parser.add_argument("output_path", type=str, help=f"Path that will become the output directory")
    args = parser.parse_args()

    merge_lora_back_into_base_model(args.lora_path, args.base_model_path, args.output_path)

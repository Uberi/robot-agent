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
    assert peft_config.task_type == peft.TaskType.CAUSAL_LM, peft_config.task_type
    base_model = transformers.AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, return_dict=True, use_safetensors=True)  # the dtype should already be torch.float16 for this particular model, but we actually have to specify this explicitly because , see https://github.com/huggingface/transformers/blob/07360b6c9c9448d619a82798419ed291dfc6ac8f/src/transformers/models/llama/convert_llama_weights_to_hf.py#L259 for details
    peft_base_model = peft.PeftModel.from_pretrained(base_model, lora_path)
    peft_base_model.eval()  # switch the model over into inference mode, disabling training-specific functionality such as dropout layers
    merged_model = peft_base_model.merge_and_unload()
    del merged_model.config._name_or_path  # remove path metadata from the model config
    merged_model.save_pretrained(output_path, safe_serialization=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lora_path", type=str, help=f"Path to the directory containing adapter_model.safetensors")
    parser.add_argument("base_model_path", type=str, help=f"Path to the directory containing config.json")
    parser.add_argument("output_path", type=str, help=f"Path that will become the output directory")
    args = parser.parse_args()

    merge_lora_back_into_base_model(args.lora_path, args.base_model_path, args.output_path)

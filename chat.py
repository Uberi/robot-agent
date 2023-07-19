import argparse
import os
import json
import random

os.environ['HF_DATASETS_OFFLINE'] = '1'  # ask datasets library not to make arbitrary web requests
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # ask transformer library not to make arbitrary web requests
os.environ['BITSANDBYTES_NOWELCOME'] = '1'  # disable welcome message that bitsandbytes prints, it's unnecessary noise

import transformers


BASE_MODEL_PATH = "./llama2-direct-preference-finetuning-output/final_checkpoint_merged"  # generated as the output of running `make train-direct-preference-finetuning`
CONTEXT_WINDOW_SIZE = 2048


if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_safetensors=True)
    tokenizer.pad_token = tokenizer.eos_token  # the padding token isn't set in the included tokenizer by default (see tokenizer.special_tokens_map for existing special tokens), set it manually
    base_model = transformers.AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, load_in_8bit=True, low_cpu_mem_usage=True, use_safetensors=True)  # load in 8-bit quantized mode
    stopping_criteria = transformers.StoppingCriteriaList([lambda input_ids, scores: tokenizer.decode(input_ids[0]).endswith("\n\n### Human:")])

    prompt_so_far = ""
    print("### Human:")
    while True:
        prompt = []
        while True:
            try:
                prompt.append(input())
            except EOFError:
                break
        if not prompt:
            break
        prompt_so_far = (prompt_so_far + "\n\n### Human:\n" + "\n".join(prompt) + "\n\n### Assistant:\n")[-CONTEXT_WINDOW_SIZE:]
        tokenized_prompt = tokenizer(prompt_so_far, return_tensors="pt").to("cuda")
        print("\n### Assistant:")
        generated_token_ids = base_model.generate(
            input_ids=tokenized_prompt.input_ids,
            attention_mask=tokenized_prompt.attention_mask,
            generation_config=transformers.GenerationConfig(
                max_new_tokens=CONTEXT_WINDOW_SIZE,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                penalty_alpha=0.6, top_k=4,  # contrastive search, gives more coherent but non-repetitive outputs
            ),
            stopping_criteria=stopping_criteria,
        )[0]
        if generated_token_ids[-1] == tokenizer.eos_token_id:
            generated_token_ids = generated_token_ids[0:-1]
        output = tokenizer.decode(generated_token_ids[tokenized_prompt.input_ids.shape[1]:])
        if output.endswith("\n\n### Human:"):
            output = output[:-len("\n\n### Human:")]
        prompt_so_far = (prompt_so_far + output)[-CONTEXT_WINDOW_SIZE:]
        print(f"{output}\n\n### Human:")

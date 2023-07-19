import argparse
import os
import re
import json
import gzip
from collections import defaultdict

os.environ['HF_DATASETS_OFFLINE'] = '1'  # ask datasets library not to make arbitrary web requests
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # ask transformer library not to make arbitrary web requests
os.environ['BITSANDBYTES_NOWELCOME'] = '1'  # disable welcome message that bitsandbytes prints, it's unnecessary noise

import numpy as np
import torch
import peft
import transformers
import trl


DATA_JSON_FILES_PATH = "./huggingface_cache/datasets--Anthropic--hh-rlhf/snapshots/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/helpful-online/"  # downloaded by running `make download-datasets-and-models` in this repo
BASE_MODEL_PATH = "./llama2-supervised-finetuning-output/final_checkpoint_merged"  # generated as the output of running `make train-supervised-finetuning`
OUTPUT_DIRECTORY = "./llama2-direct-preference-finetuning-output"
RANDOMNESS_SEED = 0
BATCH_SIZE = 1  # number of samples seen per gradient update - to increase training speed, set this to the largest size that your hardware can support without running out of memory
CONTEXT_WINDOW_SIZE = 1536  # maximum length of any input to the model, used to filter out too-long data points (this doesn't have to be the same value as in supervised_finetuning.py) - to improve performance on longer prompts, set this to the largest size that your hardware can support without running out of memory
TRAINING_STEPS = 1000  # number of steps to train for (since we're using gradient_accumulation_steps=4, the model will see TRAINING_STEPS * 4 * BATCH_SIZE samples throughout the entire training run)


def prompt_formatter(example):
    chosen_prompt_so_far = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    chosen_messages, rejected_messages = re.split(r"\n\n(Human|Assistant): ", example["chosen"])[1:], re.split(r"\n\n(Human|Assistant): ", example["rejected"])[1:]
    previous_human_is_speaking = False
    for i in range(0, min(len(chosen_messages), len(rejected_messages)), 2):
        assert chosen_messages[i] == rejected_messages[i] and chosen_messages[i] in ["Human", "Assistant"], example
        human_is_speaking = chosen_messages[i] == "Human"
        if human_is_speaking == previous_human_is_speaking:
            break
        previous_human_is_speaking = human_is_speaking

        if human_is_speaking:
            assert chosen_messages[i + 1] == rejected_messages[i + 1], example
            chosen_prompt_so_far += f'\n\n### Instruction:\n{chosen_messages[i + 1]}\n\n### Response:\n'
        else:
            yield {"prompt": chosen_prompt_so_far, "responses": [chosen_messages[i + 1], rejected_messages[i + 1]], "pairs": [(0, 1)]}
            chosen_prompt_so_far += chosen_messages[i + 1]


def create_datasets(tokenizer):
    with gzip.open(os.path.join(DATA_JSON_FILES_PATH, "train.jsonl.gz"), mode="rt") as f:
        train_dataset = [example for line in f for example in prompt_formatter(json.loads(line))]
    with gzip.open(os.path.join(DATA_JSON_FILES_PATH, "test.jsonl.gz"), mode="rt") as f:
        test_dataset = [example for line in f for example in prompt_formatter(json.loads(line))]
    return train_dataset, test_dataset


def run_training(train_dataset, test_dataset, tokenizer, resume_from_checkpoint):
    base_model = transformers.AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, load_in_8bit=True, low_cpu_mem_usage=True, use_safetensors=True)  # load in 8-bit quantized mode
    peft.prepare_model_for_kbit_training(base_model)
    if resume_from_checkpoint is not None:
        checkpoint_name = os.path.join(OUTPUT_DIRECTORY, resume_from_checkpoint)
        assert os.path.exists(checkpoint_name), checkpoint_name
        peft_base_model = peft.PeftModel.from_pretrained(base_model, checkpoint_name, is_trainable=True)  # TODO: is there a way to make this error out if it isn't in safetensors format?
    else:
        peft_base_model = peft.get_peft_model(base_model, peft.LoraConfig(
            # by default, this adds LoRA adapters around Llama2's "q_proj" and "v_proj", see https://github.com/huggingface/peft/blob/5a0e19dda1048ff8caaa12970ba7574f9cdfbf76/src/peft/utils/other.py#L280 for more details
            # some other models add more adapters, such as Guanaco, which does it on every linear layer except the head (https://github.com/artidoro/qlora/blob/845188de110d8eb7c95cc8907b54d8cb2e7c01bd/qlora.py#L221), but this doesn't seem to have too noticeable a benefit compared to models that just use these default settings, like Airoboros does (https://github.com/jondurbin/FastChat/blob/5bd738586ae6a495bd73152d74969465f30d43ac/fastchat/train/train_lora.py#L51)
            r=128,  # number of LoRA attention dimension parameters - directly proportional to LoRA adapter VRAM usage, set this to the largest value your hardware can support without running out of memory
            lora_alpha=16,  # alpha parameter for LoRA, essentially scales all of the LoRA weights, which determines "how much of an effect" this LoRA has on the final model
            lora_dropout=0.05,  # dropout probability for LoRA layers
            task_type=peft.TaskType.CAUSAL_LM,
        ))
    peft_base_model.print_trainable_parameters()
    torch.cuda.empty_cache()  # helps reduce VRAM usage - it's right after the PEFT version of the model was created, so some old stuff is still around unnecessarily

    trainer = trl.DPOTrainer(
        peft_base_model,
        base_model,  # use the original base model for comparison purposes (this model will be used in evaluation/inference mode, as a reference)
        args=transformers.TrainingArguments(
            output_dir=OUTPUT_DIRECTORY,
            dataloader_drop_last=True,  # when the dataset size isn't evenly divisible by the batch size, the remainder forms an incomplete batch - throw away this batch to avoid having to ever see an incomplete batch
            max_steps=TRAINING_STEPS,  # perform a fixed number of training steps before stopping
            evaluation_strategy="steps", eval_steps=1,  # run an evaluation every 200 training steps (~1 hour)
            save_strategy="steps", save_steps=200, save_safetensors=True,  # save a checkpoint every 200 training steps (~1 hour)
            logging_strategy="steps", logging_steps=1,  # log output every training step
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=1e-5, warmup_steps=150,  # linearly ramp up the learning rate for the AdamW optimizer from 0 to 1e-5 over the first 150 steps, then keep it at 1e-5 afterwards
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,  # use gradient checkpointing to decrease VRAM usage
            bf16=True,  # use 16-bit bfloats for training instead of 32-bit floats in most operations (some are still kept in 32-bit for precision) to decrease VRAM usage and increase training performance, in practice the precision loss has a relatively small effect on the final result
            tf32=True,  # in newer NVIDIA hardware, this replaces the remaining 32-bit operations with a 19-bit TensorFloat operations to increase training performance, in practice the precision loss has no noticeable effect on the final result
            run_name="llama2-direct-preference-finetuning",
            # TODO: when Apex becomes more stable + easier to install, look into using adamw_apex_fused rather than adamw_hf for the optim= parameter (note that some code uses optimizers= on the DPOTrainer itself, which overrides the optim= parameter on TrainingArguments, see https://github.com/huggingface/transformers/blob/53e1f5cf66d320b9c809f3940c707b6fef435d2d/src/transformers/trainer.py#L1084)
        ),
        beta=0.1,  # parameter controlling the deviation from the reference model, higher values prevent the model from deviating too far from the reference model, 0.1 is a relatively low value so the model will change quite a bit
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        max_length=CONTEXT_WINDOW_SIZE,  # any inputs that are longer than this value get truncated - first by cutting off the start of the prompt, then by cutting off the end of the response
        max_prompt_length=CONTEXT_WINDOW_SIZE // 2,  # when truncating by cutting off the start of the prompt, cut it down to half the context window size
    )

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    trainer.train()
    peft_base_model.save_pretrained(os.path.join(OUTPUT_DIRECTORY, "final_checkpoint"), safe_serialization=True)  # save LoRA by itself


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help=f"If specified, start the training from the specified checkpoint (e.g., checkpoint-500). You can get a list of all checkpoints by running: ls {OUTPUT_DIRECTORY}'")
    args = parser.parse_args()

    transformers.set_seed(RANDOMNESS_SEED)

    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_safetensors=True)
    tokenizer.pad_token = tokenizer.eos_token  # the padding token isn't set in the included tokenizer by default (see tokenizer.special_tokens_map for existing special tokens), set it manually
    train_dataset, test_dataset = create_datasets(tokenizer)
    run_training(train_dataset, test_dataset, tokenizer, args.resume_from_checkpoint)

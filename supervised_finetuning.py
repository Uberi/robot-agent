import argparse
import os
import json
import random

os.environ['HF_DATASETS_OFFLINE'] = '1'  # ask datasets library not to make arbitrary web requests
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # ask transformer library not to make arbitrary web requests
os.environ['BITSANDBYTES_NOWELCOME'] = '1'  # disable welcome message that bitsandbytes prints, it's unnecessary noise

import torch
import peft
import transformers
import trl.trainer.utils


DATA_JSON_PATH = "./huggingface_cache/datasets--c-s-ale--alpaca-gpt4-data/snapshots/88ef836e16a1d6c0658490cc521184c269c46449/data/alpaca_gpt4_data.json"  # downloaded by running `make download-datasets-and-models` in this repo
MODEL_PATH = "./huggingface_cache/models--huggyllama--llama-13b/snapshots/bf57045473f207bb1de1ed035ace226f4d9f9bba/"  # downloaded by running `make download-datasets-and-models` in this repo
OUTPUT_DIRECTORY = "./llama-supervised-finetuning-output"
RANDOMNESS_SEED = 0
BATCH_SIZE = 1  # number of samples seen per gradient update - to increase training speed, set this to the largest size that your hardware can support without running out of memory
CONTEXT_WINDOW_SIZE = 2048  # size of individual dataset entries used when training the model - to improve performance on longer prompts, set this to the largest size that your hardware can support without running out of memory
TRAINING_STEPS = 4500  # number of steps to train for (since we're using gradient_accumulation_steps=4, the model will see TRAINING_STEPS * 4 * BATCH_SIZE samples throughout the entire training run)


def prompt_formatter(example):
    if example["input"]:
        return f'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{example["instruction"]}\n\n### Input:\n{example["input"]}\n\n### Response:\n{example["output"]}'
    return f'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{example["instruction"]}\n\n### Response:\n{example["output"]}'


def create_datasets(tokenizer):
    # generate train/test split with 0.5% test
    with open(DATA_JSON_PATH) as f:
        dataset = json.load(f)
    random.shuffle(dataset)
    test_dataset_size = round(len(dataset) * 0.005)
    train_dataset, test_dataset = dataset[test_dataset_size:], dataset[:test_dataset_size]
    print(f"train dataset size: {len(train_dataset)}, test dataset size: {len(test_dataset)}")

    # estimate the average number of characters per token in the dataset using 400 samples
    total_characters, total_tokens = 0, 0
    for _, example in zip(range(400), train_dataset):
        text = prompt_formatter(example)
        total_characters += len(text)
        total_tokens += len(tokenizer(text).tokens())
    estimated_chars_per_token = total_characters / total_tokens
    print(f"dataset character to token ratio: {estimated_chars_per_token}")

    # pack multiple short examples into a single CONTEXT_WINDOW_SIZE-token-long input sequence, rather than training on each short example individually - improves training efficiency (this technique is known as "example packing")
    train_dataset_packed = trl.trainer.utils.ConstantLengthDataset(
        tokenizer,
        train_dataset,
        formatting_func=prompt_formatter,
        seq_length=CONTEXT_WINDOW_SIZE,
        chars_per_token=estimated_chars_per_token,
    )
    test_dataset_packed = trl.trainer.utils.ConstantLengthDataset(
        tokenizer,
        test_dataset,
        formatting_func=prompt_formatter,
        seq_length=CONTEXT_WINDOW_SIZE,
        chars_per_token=estimated_chars_per_token,
    )
    print(f"packed train dataset size: {sum(1 for _ in train_dataset_packed)}, packed test dataset size: {sum(1 for _ in test_dataset_packed)}")
    train_dataset_packed.infinite = True  # generate unlimited sequences by repeatedly going through the dataset
    return train_dataset_packed, test_dataset_packed


def run_training(train_dataset: torch.utils.data.IterableDataset, test_dataset: torch.utils.data.IterableDataset, resume_from_checkpoint: str):
    lora_config = peft.LoraConfig(
        # by default, this adds LoRA adapters around LLaMa's "q_proj" and "v_proj", see https://github.com/huggingface/peft/blob/5a0e19dda1048ff8caaa12970ba7574f9cdfbf76/src/peft/utils/other.py#L280 for more details
        r=128,  # number of LoRA attention dimension parameters - directly proportional to LoRA adapter VRAM usage, set this to the largest value your hardware can support without running out of memory
        lora_alpha=16,  # alpha parameter for LoRA, essentially scales all of the LoRA weights, which determines "how much of an effect" this LoRA has on the final model
        lora_dropout=0.05,  # dropout probability for LoRA layers
        task_type=peft.TaskType.CAUSAL_LM,
    )

    base_model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH, load_in_8bit=True, low_cpu_mem_usage=True, use_safetensors=True)  # load in 8-bit quantized mode
    peft.prepare_model_for_kbit_training(base_model)
    peft_base_model = peft.get_peft_model(base_model, lora_config)
    peft_base_model.print_trainable_parameters()
    torch.cuda.empty_cache()  # helps reduce VRAM usage - it's right after the PEFT version of the model was created, so some old stuff is still around unnecessarily

    if resume_from_checkpoint is not None:
        checkpoint_name = os.path.join(OUTPUT_DIRECTORY, resume_from_checkpoint, "adapter_model.bin")
        assert os.path.exists(checkpoint_name), checkpoint_name
        adapters_weights = torch.load(checkpoint_name)
        peft.set_peft_model_state_dict(peft_base_model, adapters_weights)

    trainer = transformers.Trainer(
        model=peft_base_model,
        args=transformers.TrainingArguments(
            output_dir=OUTPUT_DIRECTORY,
            dataloader_drop_last=True,  # when the dataset size isn't evenly divisible by the batch size, the remainder forms an incomplete batch - throw away this batch to avoid having to ever see an incomplete batch
            max_steps=TRAINING_STEPS,  # perform a fixed number of training steps before stopping
            evaluation_strategy="steps", eval_steps=300,  # run an evaluation every 300 training steps (~1 hour)
            save_strategy="steps", save_steps=300, save_safetensors=True,  # save a checkpoint every 300 training steps (~1 hour)
            logging_strategy="steps", logging_steps=1,  # log output every training step
            per_device_train_batch_size=BATCH_SIZE,  # batch size used in training
            per_device_eval_batch_size=BATCH_SIZE,  # batch size used in evaluation
            learning_rate=1e-5, warmup_steps=100,  # linearly ramp up the learning rate for the AdamW optimizer from 0 to 1e-5 over the first 100 steps, then keep it at 1e-5 afterwards
            gradient_accumulation_steps=4,  # use gradient accumulation to multiply effective batch size by 4 (without increasing VRAM usage by 4)
            gradient_checkpointing=True,  # use gradient checkpointing to decrease VRAM usage
            bf16=True,  # use 16-bit bfloats for training instead of 32-bit floats in most operations (some are still kept in 32-bit for precision) to decrease VRAM usage and increase training performance, in practice the precision loss has a relatively small effect on the final result
            tf32=True,  # in newer NVIDIA hardware, this replaces the remaining 32-bit operations with a 19-bit TensorFloat operations to increase training performance, in practice the precision loss has no noticeable effect on the final result
            weight_decay=0.05,  # set the weight decay regularization factor of the optimizer
            run_name="llama-supervised-finetuning",
            # TODO: when Apex becomes more stable + easier to install, look into using adamw_apex_fused rather than adamw_hf for the optim= parameter
        ),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        callbacks=[trl.trainer.utils.PeftSavingCallback],
    )

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    trainer.train()
    peft_base_model.save_pretrained(os.path.join(OUTPUT_DIRECTORY, "final_checkpoint"), safe_serialization=True)  # save LoRA by itself


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help=f"If specified, start the training from the specified checkpoint (e.g., checkpoint-500). You can get a list of all checkpoints by running: ls {OUTPUT_DIRECTORY}'")
    args = parser.parse_args()

    transformers.set_seed(RANDOMNESS_SEED)

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, use_safetensors=True)
    tokenizer.pad_token = tokenizer.eos_token  # the padding token isn't set in the included tokenizer by default (see tokenizer.special_tokens_map for existing special tokens), set it manually
    train_dataset, test_dataset = create_datasets(tokenizer)
    run_training(train_dataset, test_dataset, args.resume_from_checkpoint)

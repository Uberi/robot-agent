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


DATA_JSON_PATH = "./huggingface_cache/datasets--jondurbin--airoboros-gpt4-1.4.1/snapshots/433c04038d724bf29a193bc3c1a48b600cc417a1/instructions.jsonl"  # downloaded by running `make download-datasets-and-models` in this repo
BASE_MODEL_PATH = "./huggingface_cache/models--NousResearch--Llama-2-13b-hf/snapshots/81da3af9503579bf991e3995564baa683b27d38c/"  # downloaded by running `make download-datasets-and-models` in this repo
OUTPUT_DIRECTORY = "./llama2-supervised-finetuning-output"
RANDOMNESS_SEED = 0
BATCH_SIZE = 1  # number of samples seen per gradient update - to increase training speed, set this to the largest size that your hardware can support without running out of memory
CONTEXT_WINDOW_SIZE = 2048  # size of individual dataset entries used when training the model - to improve performance on longer prompts, set this to the largest size that your hardware can support without running out of memory
TRAINING_STEPS = 4500  # number of steps to train for (since we're using gradient_accumulation_steps=4, the model will see TRAINING_STEPS * 4 * BATCH_SIZE samples throughout the entire training run)


def prompt_formatter(example):
    return f'\n\n### Human:\n{example["instruction"]}\n\n### Assistant:\n{example["response"]}'


def create_datasets(tokenizer):
    # generate train/test split with 0.5% test
    with open(DATA_JSON_PATH) as f:
        dataset = [prompt_formatter(json.loads(line)) for line in f]
    random.shuffle(dataset)
    test_dataset_size = round(len(dataset) * 0.005)
    train_dataset, test_dataset = dataset[test_dataset_size:], dataset[:test_dataset_size]
    print(f"train dataset size: {len(train_dataset)}, test dataset size: {len(test_dataset)}")

    # estimate the average number of characters per token in the dataset using 400 samples
    total_characters, total_tokens = 0, 0
    for _, example in zip(range(400), train_dataset):
        total_characters += len(example)
        total_tokens += len(tokenizer(example).tokens())
    estimated_chars_per_token = total_characters / total_tokens
    print(f"dataset character to token ratio: {estimated_chars_per_token}")

    # pack multiple short examples into a single CONTEXT_WINDOW_SIZE-token-long input sequence, rather than training on each short example individually - improves training efficiency (this technique is known as "example packing")
    train_dataset_packed = trl.trainer.utils.ConstantLengthDataset(
        tokenizer,
        train_dataset,
        formatting_func=lambda x: x,
        seq_length=CONTEXT_WINDOW_SIZE,
        chars_per_token=estimated_chars_per_token,
    )
    test_dataset_packed = trl.trainer.utils.ConstantLengthDataset(
        tokenizer,
        test_dataset,
        formatting_func=lambda x: x,
        seq_length=CONTEXT_WINDOW_SIZE,
        chars_per_token=estimated_chars_per_token,
    )
    print(f"packed train dataset size: {sum(1 for _ in train_dataset_packed)}, packed test dataset size: {sum(1 for _ in test_dataset_packed)}")
    train_dataset_packed.infinite = True  # generate unlimited sequences by repeatedly going through the dataset
    return train_dataset_packed, test_dataset_packed


def run_training(train_dataset: torch.utils.data.IterableDataset, test_dataset: torch.utils.data.IterableDataset, resume_from_checkpoint: str):
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
            run_name="llama2-supervised-finetuning",
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

    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_safetensors=True)
    tokenizer.pad_token = tokenizer.eos_token  # the padding token isn't set in the included tokenizer by default (see tokenizer.special_tokens_map for existing special tokens), set it manually
    train_dataset, test_dataset = create_datasets(tokenizer)
    run_training(train_dataset, test_dataset, args.resume_from_checkpoint)

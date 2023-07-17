import argparse
import os
import json
import random

os.environ['HF_DATASETS_OFFLINE'] = '1'  # ask datasets library not to make arbitrary web requests
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # ask transformer library not to make arbitrary web requests
os.environ['BITSANDBYTES_NOWELCOME'] = '1'  # disable welcome message that bitsandbytes prints, it's unnecessary noise

import numpy as np
import torch
import peft
import transformers


DATA_JSON_PATH = "./huggingface_cache/gpt-4-llm-comparison_data_v2.json"  # downloaded by running `make download-datasets-and-models` in this repo
MODEL_PATH = "./llama-supervised-finetuning-output/final_checkpoint_merged"  # generated as the output of running `make train-supervised-finetuning`
OUTPUT_DIRECTORY = "./llama-reward-model-output"
RANDOMNESS_SEED = 0
BATCH_SIZE = 1  # number of samples seen per gradient update - to increase training speed, set this to the largest size that your hardware can support without running out of memory
CONTEXT_WINDOW_SIZE = 2048  # maximum length of any input to the model, used to filter out too-long data points - set this to the same value as the corresponding CONTEXT_WINDOW_SIZE variable in supervised_finetuning.py
TRAINING_STEPS = 4500  # number of steps to train for (since we're using gradient_accumulation_steps=4, the model will see TRAINING_STEPS * 4 * BATCH_SIZE samples throughout the entire training run)


def prompt_formatter(example):
    # the comparison dataset consists of responses from GPT-4, GPT-3, and OPT-IML-1.3B, all scored out of 10 by GPT-4
    # we'll choose the best-scoring response first, breaking ties by model (e.g., GPT-4's respose will always be chosen over GPT-3's response when they both score the same)
    # then a worse-scoring response is chosen either by the next-lowest score, or the next-lower-ranking model if there is no next-lowest score
    all_scores = sorted(set(r["score"] for r in example["responses_and_scores"]))
    model_rank = {"gpt4": 1, "text-davinci-003": 2, "icm-1.3b": 3}
    best_responses = sorted((r for r in example["responses_and_scores"] if r["score"] == all_scores[-1]), key=lambda r: (r["score"], model_rank[r["source"]]))
    if len(all_scores) == 1:  # all scored the same
        better_response, worse_response = best_responses[0], best_responses[1]
    else:  # multiple different scores, find the next-lowest-score model
        better_response, worse_response = best_responses[0], min((r for r in example["responses_and_scores"] if r["score"] == all_scores[-2]), key=lambda r: model_rank[r["source"]])
    return f'{example["user_input"]}\n\n### Response:\n{better_response["response"]}', f'{example["user_input"]}\n\n### Response:\n{worse_response["response"]}'  # user_input already contains the instruction and input sections of the prompt, so we just need to add the response here


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        return iter(self.dataset)


def create_datasets(tokenizer):
    with open(DATA_JSON_PATH) as f:
        dataset = json.load(f)

    # choose the better/worse examples, tokenize them, and shuffle the resulting dataset
    processed_dataset = []
    for example in dataset:
        if len(example["responses_and_scores"]) < 2:
            continue
        better_prompt, worse_prompt = prompt_formatter(example)
        better_tokenized, worse_tokenized = tokenizer(better_prompt), tokenizer(worse_prompt)
        if len(better_tokenized["input_ids"]) > CONTEXT_WINDOW_SIZE or len(worse_tokenized["input_ids"]) > CONTEXT_WINDOW_SIZE:  # tokenized question-answer pair too long, skip this one
            continue
        processed_dataset.append({"better_input_ids": better_tokenized["input_ids"], "better_attention_mask": better_tokenized["attention_mask"], "worse_input_ids": worse_tokenized["input_ids"], "worse_attention_mask": worse_tokenized["attention_mask"]})  # store tokens (as a list of ints) and attention masks (list of bool-ints masking off token attention; usually used to mask off padding tokens so we don't do attention on them)
    random.shuffle(processed_dataset)

    # generate train/test split with 1% test
    test_dataset_size = round(len(processed_dataset) * 0.01)
    train_dataset, test_dataset = processed_dataset[test_dataset_size:], processed_dataset[:test_dataset_size]
    print(f"train dataset size: {len(train_dataset)}, test dataset size: {len(test_dataset)}")
    return Dataset(train_dataset), Dataset(test_dataset)


class RewardTrainer(transformers.Trainer):
    def compute_loss(self, model, batch, return_outputs=False):
        better_rewards = model(input_ids=batch["better_input_ids"], attention_mask=batch["better_attention_mask"])[0]
        worse_rewards = model(input_ids=batch["worse_input_ids"], attention_mask=batch["worse_attention_mask"])[0]
        loss = -torch.nn.functional.logsigmoid(better_rewards - worse_rewards).mean()  # pairwise logloss as defined in [the InstructGPT/RLHF paper](https://arxiv.org/abs/2203.02155)
        return (loss, (loss, better_rewards, worse_rewards)) if return_outputs else loss  # when return_outputs is true, the outputs format should either be a dict (in which case the values get turned into an array in insertion order, very unusual design decision), or a list/tuple (in which case the first element gets trimmed off, presumably because the library assumes that the first element is the loss and the rest of the parameters are the logits), see https://github.com/huggingface/transformers/blob/5bb4430edc7df9f9950d412d98bbe505cc4d328b/src/transformers/trainer.py#L3343 for details


def run_training(train_dataset: torch.utils.data.IterableDataset, test_dataset: torch.utils.data.IterableDataset, resume_from_checkpoint: str):
    peft_config = peft.LoraConfig(
        r=8,  # number of LoRA attention dimension parameters - directly proportional to LoRA adapter VRAM usage, set this to the largest value your hardware can support without running out of memory
        lora_alpha=32,  # alpha parameter for LoRA, essentially scales all of the LoRA weights, which determines "how much of an effect" this LoRA has on the final model
        lora_dropout=0.1,  # dropout probability for LoRA layers
        task_type=peft.TaskType.SEQ_CLS,
    )

    base_model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, load_in_8bit=True, num_labels=1, low_cpu_mem_usage=True, use_safetensors=True)  # when num_labels=1, the output of this model is the model's regression loss (mean-square loss) for any given input
    peft.prepare_model_for_kbit_training(base_model)
    peft_base_model = peft.get_peft_model(base_model, peft_config)
    peft_base_model.print_trainable_parameters()
    torch.cuda.empty_cache()  # helps reduce VRAM usage - it's right after the PEFT version of the model was created, so some old stuff is still around unnecessarily

    if resume_from_checkpoint is not None:
        checkpoint_name = os.path.join(OUTPUT_DIRECTORY, resume_from_checkpoint, "adapter_model.bin")
        assert os.path.exists(checkpoint_name), checkpoint_name
        adapters_weights = torch.load(checkpoint_name)
        peft.set_peft_model_state_dict(peft_base_model, adapters_weights)

    def compute_metrics(eval_prediction):
        better_rewards, worse_rewards = eval_prediction.predictions  # this is actually the logits, namely the second return value of RewardTrainer.compute_loss(..., return_outputs=True) with its first element removed, see https://github.com/huggingface/transformers/blob/5bb4430edc7df9f9950d412d98bbe505cc4d328b/src/transformers/trainer.py#L3727 for details
        return {"accuracy": np.sum(better_rewards > worse_rewards) / len(better_rewards)}  # the accuracy metric is how often the model predicts a higher reward for the "better" response than for the "worse" response, randomly choosing would give 50%, and perfectly choosing would give 100%

    def build_batch_from_dataset_subset(dataset_subset):
        # pad all of the better and worse examples in this subset of the dataset up to the longest example's length, and turn them into one PyTorch tensor for the better examples, and another for the worse
        # see https://huggingface.co/docs/transformers/v4.30.0/en/pad_truncation#padding-and-truncation and https://github.com/huggingface/transformers/blob/5bb4430edc7df9f9950d412d98bbe505cc4d328b/src/transformers/tokenization_utils_base.py#L2907 for more details
        better_examples = tokenizer.pad(
            [{"input_ids": example["better_input_ids"], "attention_mask": example["better_attention_mask"]} for example in dataset_subset],
            padding='longest',
            return_tensors="pt",  # return result as a PyTorch tensor
        )
        worse_examples = tokenizer.pad(
            [{"input_ids": example["worse_input_ids"], "attention_mask": example["worse_attention_mask"]} for example in dataset_subset],
            padding='longest',
            return_tensors="pt",  # return result as a PyTorch tensor
        )
        return {
            "better_input_ids": better_examples["input_ids"],
            "better_attention_mask": better_examples["attention_mask"],
            "worse_input_ids": worse_examples["input_ids"],
            "worse_attention_mask": worse_examples["attention_mask"],
            "dummy_label": torch.tensor([0]),  # unnecessary label since there's only one label actually, but if not present then trainer.prediction_step() will think there are no labels and won't actually call trainer.compute_loss(), it's a dirty hack to get around all the reflection and "magic" functionality in this library; also, because of a recent bug introduced, the value has to be a torch.Tensor: https://github.com/huggingface/accelerate/issues/1611
        }

    trainer = RewardTrainer(
        model=peft_base_model,
        args=transformers.TrainingArguments(
            output_dir=OUTPUT_DIRECTORY,
            dataloader_drop_last=True,  # when the dataset size isn't evenly divisible by the batch size, the remainder forms an incomplete batch - throw away this batch to avoid having to ever see an incomplete batch
            max_steps=TRAINING_STEPS,  # perform a fixed number of training steps before stopping
            evaluation_strategy="steps", eval_steps=400,  # run an evaluation every 400 training steps (~30 minutes)
            save_strategy="steps", save_steps=800, save_safetensors=True,  # save a checkpoint every 800 training steps (~1 hour)
            logging_strategy="steps", logging_steps=1,  # log output every training step
            per_device_train_batch_size=BATCH_SIZE,  # batch size used in training
            per_device_eval_batch_size=BATCH_SIZE,  # batch size used in evaluation
            learning_rate=1e-5, warmup_steps=100,  # linearly ramp up the learning rate for the AdamW optimizer from 0 to 1e-5 over the first 100 steps, then keep it at 1e-5 afterwards
            gradient_accumulation_steps=4,  # use gradient accumulation to multiply effective batch size by 4 (without increasing VRAM usage by 4)
            gradient_checkpointing=True,  # use gradient checkpointing to decrease VRAM usage
            bf16=True,  # use 16-bit bfloats for training instead of 32-bit floats in most operations (some are still kept in 32-bit for precision) to decrease VRAM usage and increase training performance, in practice the precision loss has a relatively small effect on the final result
            tf32=True,  # in newer NVIDIA hardware, this replaces the remaining 32-bit operations with a 19-bit TensorFloat operations to increase training performance, in practice the precision loss has no noticeable effect on the final result
            weight_decay=0.001,
            run_name="llama-reward-model",
            remove_unused_columns=False,  # by default, the transformers library removes any column from the dataset that doesn't match a name that's a parameter name of the .forward() method of the model, we don't want that because our dataset has other important fields like "better_input_ids"
            label_names=["dummy_label"],  # by default, label names for sequence classification models are taken from the parameter names in the .forward() method of the model containing "label" (see https://github.com/huggingface/transformers/blob/5bb4430edc7df9f9950d412d98bbe505cc4d328b/src/transformers/trainer.py#L687 for details), which is meaningless here, so even though there's only one label we still have to set one
            # TODO: when Apex becomes more stable + easier to install, look into using adamw_apex_fused rather than adamw_hf for the optim= parameter
        ),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=build_batch_from_dataset_subset,
    )

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    trainer.train()
    peft_base_model.save_pretrained(os.path.join(OUTPUT_DIRECTORY, "final_checkpoint"), safe_serialization=True)
    reward_model = peft_base_model.merge_and_unload()  # merge the LoRA back into the base model
    reward_model.save_pretrained(os.path.join(OUTPUT_DIRECTORY, "final_checkpoint"), safe_serialization=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help=f"If specified, start the training from the specified checkpoint (e.g., checkpoint-500). You can get a list of all checkpoints by running: ls {OUTPUT_DIRECTORY}'")
    args = parser.parse_args()

    transformers.set_seed(RANDOMNESS_SEED)

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, use_safetensors=True)
    tokenizer.pad_token = tokenizer.eos_token  # the padding token isn't set in the included tokenizer by default (see tokenizer.special_tokens_map for existing special tokens), set it manually
    train_dataset, test_dataset = create_datasets(tokenizer)
    run_training(train_dataset, test_dataset, args.resume_from_checkpoint)

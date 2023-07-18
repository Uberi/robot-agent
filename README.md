Robot Agent
===========

Fine-tuned 13B LLaMa model designed for ReAct-style and Tree-Of-Thoughts style prompting. The codebase has the following desirable features:

* Entire training procedure runs out of the box on a single RTX 4090 with less than 30 hours of compute time.
    * Requires 22.7GiB of the available 24GiB of VRAM.
    * This is accomplished through tuning, quantization, FP16, TF32, and the usual gradient accumulation/checkpointing settings.
    * Training is fully interruptible - will automatically resume from latest checkpoint.
* Heavily commented, short, clean, and reproducible training code.
    * All library dependency versions fully pinned, base models pinned and downloaded as part of setup process.
    * After initial setup, training process does not require network access.
    * Avoids leaving cache files throughout the user's system, minimizes extraneous files whereever practical.
    * Use SafeTensors everywhere for speed and security.

Technical details:

* QLoRA training, a 128 rank LoRA similar to [Guanaco](https://github.com/artidoro/qlora/blob/cc488110b5ea23594a418daca7085000a9420625/qlora.py#L324).
* Full 2048-token context window used in training. TODO: not in the DPO part of it yet, need to get VRAM usage down a bit more
* Finetuning using [Alpaca's GPT-4 instruction-following dataset](https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data).
* DPO using [Anthropic's hh-rlhf dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf).
* Based on LLaMa 13B.
* Codebase takes ideas and inspiration from [StackLLaMa](https://github.com/lvwerra/trl/tree/5c7bfbc8d9aeabee893290cc02121d7260636978/examples/research_projects/stack_llama/scripts), [QLoRA](https://github.com/artidoro/qlora), and [LLaMa-TRL](https://github.com/jasonvanf/llama-trl).

Prompt Format
-------------

TODO: switch over to Vicuna format

With input:

```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
INSTRUCTIONS_GO_HERE

### Input:
INPUT_GOES_HERE

### Response:
```

Without input:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
INSTRUCTIONS_GO_HERE

### Response:
```

Note that both prompt formats end in a single newline, and have no preceding whitespace.

Training
--------

```
make download-datasets-and-models  # this step requires an internet connection
make train  # this step can be run fully offline, including on airgapped systems, as long as the entire project folder is transferred over
```

Inference
---------

TODO: write this up, maybe export some GGMLs and GPTQs

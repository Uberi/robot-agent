Robot Agent
===========

Fine-tuned 13B LLaMa model designed for ReAct-style and Tree-Of-Thoughts style prompting. The codebase has the following desirable features:

* Entire training procedure runs out of the box on a single RTX 4090 with less than 60 hours of compute time. TODO: get more detailed info, seems to vary
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
* Full 2048-token context window used in training.
* Based on LLaMa 13B.
* Codebase takes ideas and inspiration from [StackLLaMa](https://github.com/lvwerra/trl/tree/5c7bfbc8d9aeabee893290cc02121d7260636978/examples/research_projects/stack_llama/scripts), [QLoRA](https://github.com/artidoro/qlora), and [LLaMa-TRL](https://github.com/jasonvanf/llama-trl).

Prompt Format
-------------

TODO

Training
--------

```
make download-datasets-and-models
make train-supervised-finetuning
```

TODO

Inference
---------

TODO

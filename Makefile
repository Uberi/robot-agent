#######################
# user-facing targets #
#######################

.PHONY: train
train: download-datasets-and-models train-supervised-finetuning train-direct-preference-finetuning

.PHONY: download-datasets-and-models
download-datasets-and-models: venv/requirements_installed
	. ./venv/bin/activate && python3 -c 'import huggingface_hub as h; h.hf_hub_download(repo_id="jondurbin/airoboros-gpt4-1.4.1", repo_type="dataset", revision="433c04038d724bf29a193bc3c1a48b600cc417a1", filename="instructions.jsonl", cache_dir="./huggingface_cache", resume_download=True)'
	. ./venv/bin/activate && python3 -c 'import huggingface_hub as h; h.snapshot_download(repo_id="NousResearch/Llama-2-13b-hf", revision="81da3af9503579bf991e3995564baa683b27d38c", ignore_patterns=["pytorch_*"], cache_dir="./huggingface_cache", resume_download=True)'
	. ./venv/bin/activate && python3 -c 'import huggingface_hub as h; h.snapshot_download(repo_id="Anthropic/hh-rlhf", repo_type="dataset", revision="09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa", cache_dir="./huggingface_cache", resume_download=True)'

.PHONY: train-supervised-finetuning
train-supervised-finetuning: llama2-supervised-finetuning-output/final_checkpoint_merged

.PHONY: train-reward-modeling
train-direct-preference-finetuning: llama2-direct-preference-finetuning-output/final_checkpoint_merged

.PHONY: chat
chat: train-direct-preference-finetuning
	. ./venv/bin/activate && python3 chat.py

.PHONY: generate-ggml
generate-ggml: train-direct-preference-finetuning
	# TODO: download llama.cpp and build it
	mkdir -p exported-models
	. ./venv/bin/activate && cd llama.cpp && python3 convert.py --outfile ../exported-models/ggml-robot-agent-f16.bin ../llama2-direct-preference-finetuning-output/final_checkpoint_merged && ./quantize ../exported-models/ggml-robot-agent-f16.bin ../exported-models/ggml-robot-agent-q5_K_M.bin q5_K_M

llama-cpp-chat:
	cd llama.cpp && ./main --model ../exported-models/ggml-robot-agent-q5_K_M.bin --color -i --interactive-first --mirostat 2 --ctx-size 2048 -r $$'\n\n### Human:\n' --in-prefix $$'\n\n### Human:\n' --in-suffix $$'\n\n### Assistant:\n' -n -1

####################
# internal targets #
####################

llama2-supervised-finetuning-output/final_checkpoint_merged: venv/requirements_installed
	. ./venv/bin/activate && python3 supervised_finetuning.py
	. ./venv/bin/activate && python3 merge_peft_adapter.py llama2-supervised-finetuning-output/final_checkpoint huggingface_cache/models--NousResearch--Llama-2-13b-hf/snapshots/81da3af9503579bf991e3995564baa683b27d38c llama2-supervised-finetuning-output/final_checkpoint_merged

llama2-direct-preference-finetuning-output/final_checkpoint_merged: venv/requirements_installed llama2-supervised-finetuning-output/final_checkpoint_merged
	. ./venv/bin/activate && python3 direct_preference_finetuning.py
	. ./venv/bin/activate && python3 merge_peft_adapter.py llama2-direct-preference-finetuning-output/final_checkpoint llama2-supervised-finetuning-output/final_checkpoint_merged llama2-direct-preference-finetuning-output/final_checkpoint_merged

venv/requirements_installed: requirements.txt
	python3 -m venv venv
	. ./venv/bin/activate && pip install -r requirements.txt && touch ./venv/requirements_installed

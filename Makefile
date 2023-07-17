#######################
# user-facing targets #
#######################

.PHONY: train-supervised-finetuning
train-supervised-finetuning: llama-supervised-finetuning-output/final_checkpoint_merged

.PHONY: train-reward-modeling
train-reward-modeling: llama-reward-modeling-output/final_checkpoint_merged

.PHONY: download-datasets-and-models
download-datasets-and-models: venv/requirements_installed
	. ./venv/bin/activate && python3 -c 'import huggingface_hub as h; h.hf_hub_download(repo_id="c-s-ale/alpaca-gpt4-data", repo_type="dataset", revision="88ef836e16a1d6c0658490cc521184c269c46449", filename="data/alpaca_gpt4_data.json", cache_dir="./huggingface_cache", resume_download=True)'
	. ./venv/bin/activate && python3 -c 'import huggingface_hub as h; h.snapshot_download(repo_id="huggyllama/llama-13b", revision="bf57045473f207bb1de1ed035ace226f4d9f9bba", ignore_patterns=["pytorch_*"], cache_dir="./huggingface_cache", resume_download=True)'
	wget --continue -O ./huggingface_cache/gpt-4-llm-comparison_data_v2.json https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/raw/c242c50660f678f63ab79d7f50e492a30de1b5a6/data/comparison_data_v2.json

####################
# internal targets #
####################

llama-supervised-finetuning-output/final_checkpoint_merged: venv/requirements_installed
	. ./venv/bin/activate && python3 supervised_finetuning.py
	. ./venv/bin/activate && python3 merge_peft_adapter.py llama-supervised-finetuning-output/final_checkpoint huggingface_cache/models--huggyllama--llama-13b/snapshots/bf57045473f207bb1de1ed035ace226f4d9f9bba llama-supervised-finetuning-output/final_checkpoint_merged

llama-reward-modeling-output/final_checkpoint_merged: venv/requirements_installed llama-supervised-finetuning-output/final_checkpoint_merged
	. ./venv/bin/activate && python3 reward_modeling.py
	. ./venv/bin/activate && python3 merge_peft_adapter.py llama-reward-modeling-output/final_checkpoint llama-supervised-finetuning-output/final_checkpoint_merged llama-reward-modeling-output/final_checkpoint_merged

venv/requirements_installed: requirements.txt
	python3 -m venv venv
	. ./venv/bin/activate && pip install -r requirements.txt && touch ./venv/requirements_installed

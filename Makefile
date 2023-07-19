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
generate-ggml: exported-models/ggml-robot-agent-q5_K_M.bin

chat-llama-cpp: exported-models/ggml-robot-agent-q5_K_M.bin llama.cpp/main
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
	. ./venv/bin/activate && pip install setuptools==68.0.0 && pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118  # install the CUDA 11.8 version of PyTorch rather than the default CUDA 11.7 version for a nice 50% GPU performance bump, currently the PyTorch install page (https://pytorch.org/get-started/locally/) shows "pip install torch" under the CUDA 11.7 section, whereas the CUDA 11.8 section shows the different command we're using here. because it's using a different Python package index, we also can't put this in the requirements.txt file either
	. ./venv/bin/activate && pip install -r requirements.txt && touch ./venv/requirements_installed

llama.cpp/main:
	[ ! -d llama.cpp/.git ] && git clone https://github.com/ggerganov/llama.cpp.git
	cd llama.cpp && git reset --hard d01bccde9f759b24449fdaa16306b406a50eb367 && make

exported-models/ggml-robot-agent-q5_K_M.bin: llama2-direct-preference-finetuning-output/final_checkpoint_merged llama.cpp/main
	mkdir -p exported-models
	. ./venv/bin/activate && python3 llama.cpp/convert.py --outfile exported-models/ggml-robot-agent-f16.bin llama2-direct-preference-finetuning-output/final_checkpoint_merged
	./llama.cpp/quantize exported-models/ggml-robot-agent-f16.bin exported-models/ggml-robot-agent-q5_K_M.bin q5_K_M

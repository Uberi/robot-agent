.PHONY: train-supervised-finetuning
train-supervised-finetuning: venv/requirements_installed
	. ./venv/bin/activate && python3 supervised_finetuning.py

.PHONY: download-datasets-and-models
download-datasets-and-models: venv/requirements_installed
	. ./venv/bin/activate && python3 -c 'import huggingface_hub as h; h.hf_hub_download(repo_id="c-s-ale/alpaca-gpt4-data", repo_type="dataset", revision="88ef836e16a1d6c0658490cc521184c269c46449", filename="data/alpaca_gpt4_data.json", cache_dir="./huggingface_cache")'
	. ./venv/bin/activate && python3 -c 'import huggingface_hub as h; h.snapshot_download(repo_id="huggyllama/llama-13b", revision="bf57045473f207bb1de1ed035ace226f4d9f9bba", ignore_patterns=["pytorch_*"], cache_dir="./huggingface_cache")'

.PHONY: clean
clean:
	rm -rf ./venv ./huggingface_cache

venv/requirements_installed: requirements.txt
	python3 -m venv venv
	. ./venv/bin/activate && pip install -r requirements.txt && touch ./venv/requirements_installed

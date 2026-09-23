# hua

[hua-1.7b](https://huggingface.co/neelabhbuilds/hua-1.7b) ("hua" = home use agent) is Qwen3-1.7B trained with reinforcement learning (GRPO) to control Home Assistant through its tool calls.

This repo is the server that runs it for Home Assistant, and the files to reproduce its scores:

- `serve.py` - an OpenAI-compatible chat-completions server for Home Assistant's core `llama_cpp` integration. It renders the incoming messages and tools with Qwen3's chat template (thinking off) and returns the model's `<tool_call>` blocks as OpenAI `tool_calls`. This is the server the published scores were produced with.
- `parse.py` - reads the tool calls out of the model's answer.
- `harness/hua-1.7b.yaml` - a model entry for the benchmark harness.

## Requirements

- Python 3.12, `pip install -r requirements.txt` (torch 2.14.0, transformers 5.17.0; the torch wheel used was the CUDA 13.0 build).
- An NVIDIA GPU with about 4 GB free. The weights are bf16 (3.4 GB) and are served unquantized; peak memory during the benchmark runs was 3.9 GiB on an RTX 3060. The model has only been measured at bf16.

## Run the server

```bash
python serve.py
```

This downloads `neelabhbuilds/hua-1.7b` from Hugging Face and listens on `0.0.0.0:8017`. Use `--model-path /path/to/hua-1.7b` for a local copy of the weights, `--port` to change the port. The server answers to the model name `hua-1.7b` (`--model-name` to change it) and refuses requests for any other name. Requests are served one at a time.

Check it is up:

```bash
curl -s localhost:8017/v1/models
```

## Connect Home Assistant

In Home Assistant: Settings > Devices & services > Add integration > **llama.cpp** (the core `llama_cpp` integration). It asks for:

1. **URL**: `http://<host>:8017/v1`, where `<host>` is the machine running `serve.py`. **API key** is optional; `serve.py` does not check it, so leave it empty or put anything.
2. **Model**: pick `hua-1.7b` from the list (it comes from the server's `/v1/models`).

Home Assistant then sends a short test request, and a second one with streaming, and creates a conversation agent with "Recommended model settings" on and "Control Home Assistant" set to Assist. The recommended settings are temperature 0.7, top P 1.0 and 3000 max tokens, which are the settings the scores were run with. Leave the instructions at their default. Then choose the agent as the conversation agent of a voice assistant (Settings > Voice assistants).

Because `serve.py` answers the streaming test, Home Assistant turns streaming on. The scores were run with streaming off. `serve.py` writes the whole answer before it streams it, so the answer is the same either way.

## Reproduce the numbers

Results from the model card, on three datasets from [home-assistant-datasets](https://github.com/allenporter/home-assistant-datasets):

| model | assist-mini (196) | assist (460) | questions (370) |
|---|---|---|---|
| Qwen3-1.7B, stock, bf16 | 75.0 % ± 6.1 (147) | 51.5 % ± 4.6 (237) | 68.4 % ± 4.7 (253) |
| **hua-1.7b v1**, bf16 | **89.8 % ± 4.2 (176)** | **68.0 % ± 4.3 (313)** | **70.0 % ± 4.7 (259)** |

Setup for both rows: the harness at commit `08315410753adf60a49904b498bc8046fb2e4ea6`, Home Assistant 2026.9.3, bf16 weights served by this `serve.py` (default settings) on one RTX 3060 12 GB, connected through the `llama_cpp` integration, temperature 0.7. Each result is the share of tasks where the whole home ends in the expected state, with the reply checked where the task expects one; the ± is the harness's error bar and the count in brackets is tasks passed. The temperature is 0.7, so a rerun moves within the error bar.

The harness needs Python 3.14 (Home Assistant 2026.9.3 requires it) and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/allenporter/home-assistant-datasets
cd home-assistant-datasets
git checkout 08315410753adf60a49904b498bc8046fb2e4ea6
git clone https://github.com/allenporter/home-assistant-synthetic-home
uv venv --python 3.14 .venv
source .venv/bin/activate
uv pip install -r requirements_dev.txt --prerelease=allow
uv pip install -r requirements_eval.txt --prerelease=allow
uv pip freeze | grep '^homeassistant=='      # 2026.9.3
cp /path/to/hua/harness/hua-1.7b.yaml models/
```

Start `python serve.py` on the same machine (the model entry points at `http://localhost:8017/v1`). Then, for each dataset (`assist-mini`, `assist`, `questions`), collect the answers and score them:

```bash
DS=assist-mini
OUT=reports/$DS/2026.9.3
mkdir -p $OUT
pytest home_assistant_datasets/tool/assist/collect \
    --models=hua-1.7b --dataset=datasets/$DS/ --model_output_dir=$OUT
# the eval reports every miss as a failed test, so it exits 1 whenever anything was missed
pytest home_assistant_datasets/tool/assist/eval --model_output_dir=$OUT || true
cat $OUT/reports.yaml
```

`reports.yaml` has the score, the error bar and the count; `report.csv` has one row per task. On an RTX 3060 the collect took about 7 minutes for assist-mini, 20 for assist and 18 for questions.

For the stock row, run the same server on the base weights, `python serve.py --model-path Qwen/Qwen3-1.7B --model-name qwen3-1.7b-bf16`, with a copy of the model entry whose `model_id` and `chat_model` are `qwen3-1.7b-bf16`.

## License

Apache-2.0, see `LICENSE`. The model is built on [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) by the Qwen team, also Apache-2.0.

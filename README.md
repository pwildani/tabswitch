# tabswitch

A simple CLI for loading/unloading models for tabbyAPI.

This includes functionality for accessing tabbyAPI's API, and can be run remotely on a separate system.

This does not provide an LLM inference frontend - use any compatible frontend of your choice.


## Prerequisites
To get started, make sure you have the following installed on your system:

* Python 3.10+
* The `requests` library (installable via pip or system packages or whatever)

## Installation
1. Clone this repository to your machine: git clone https://github.com/pwildani/tabswitch
1. cd into the new directory.
1. Set up a config file named `tabswitch/config.yml` in your config directory (so $HOME/.config on Linux)
1. Either:
   * Ensure `requests` is installed and put tabswitch.py wherever you want to run it from
   * `pip install .`

## Usage

```
$ tabswitch --help
usage: tabswitch [-h] [--ctx MAX_SEQ_LEN] [-T PROMPT_TEMPLATE] [--cache-mode {Q4,FP8,FP16}]
                 [mode] [modelword ...]

positional arguments:
  mode
  modelword

options:
  -h, --help            show this help message and exit
  --ctx MAX_SEQ_LEN, --context-length MAX_SEQ_LEN
  -T PROMPT_TEMPLATE, --prompt-template PROMPT_TEMPLATE
  --cache-mode {Q4,FP8,FP16}

$ tabswitch help
current, list, set-model <exact name>, model <approximate name>,
select-model <approximate name>, or just <approximate name>

$ tabswitch select-model gemma
Multiple matching models:
gemma-2-27b-it-4.5bpw-exl2
gemma-2-27b-it-5.25bpw-exl2
gemma-2-27b-it-5.65bpw-exl2
gemma-2-27b-it-5bpw-exl2

$ tabswitch gemma 5
gemma-2-27b-it-5bpw-exl2
100%|###| 95/95 [00:14<00:00,  6.78 Layers/s]

$ tabswitch nonesuch
No matching models

$ tabswitch current
{'id': 'gemma-2-27b-it-5bpw-exl2', 'object': 'model', 'created': 1722373156,
'owned_by': 'tabbyAPI', 'logging': {'prompt': True, 'generation_params': True},
'parameters': {'max_seq_len': 8192, 'rope_scale': 1.0, 'rope_alpha': 1.0,
'cache_size': 8192, 'cache_mode': 'Q4', 'chunk_size': 2048, 'prompt_template':
'from_tokenizer_config', 'num_experts_per_token': None, 'draft': None}}
```

Selecting a model by approximate name will find models with names that match
the name fragments on the commmand line.


## Example configuration
In: `$HOME/.config/tabswitch/config.ini`

```ini
api_root: http://localhost:5000/
api_key: abcde
admin_key: 12435
```

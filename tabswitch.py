#!/usr/bin/python3
from collections import defaultdict
from pathlib import Path
from typing import TypedDict, Protocol
import argparse
import configparser
import json
import os
import re
import sys
import typing

import requests


try:
    from tqdm import tqdm
except ImportError:
    # tqdm progress bar not installed, fake it out.
    from unittest.mock import MagicMock

    tqdm = MagicMock()


T = typing.TypeVar("T")

# Couldn't get zero width split working.
#LOWER_UPPER_TRANSITION = regex.compile("(?<=[a-z_])(?=[0-9A-Z])", flags=regex.VERSION1)

# Matches a word terminated by a lower-upper case transition,
# or lower-digit transition, or whitespace
LU_WORD = re.compile(
    r"""
        (?# Leading)
        [0-9A-Z.]*

        (?# lower word)
        [^0-9A-Z\s]+

        (?# Transition to upper/digits/period or end of string)
        (?:
             (?=[0-9A-Z\s])
           |
             $
        )
    |
        (?# Or just match all terminals )
        [0-9A-Z]+
""",
    flags=re.VERBOSE,
)


class ModelConfig(Protocol):
    cache_mode: str | None
    prompt_template: str | None
    max_seq_len: int | None


class EncodeTokensResponse(TypedDict):
    tokens: list[int]
    length: int


class SamplerSettings(TypedDict, total=False):
    max_tokens: int = 150
    min_tokens: int = 0
    generate_window: int = 512
    stop: str | list[str]
    banned_strings: str | None = None
    token_healing: bool = True
    temperature: float = 1.0
    temperature_last: bool = True
    smoothing_factor: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    top_a: float = 0.0
    min_p: float = 0.0
    tfs: float = 1.0
    skew: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    repetition_decay: float = 0.0
    mirostat_mode: int = 0
    mirostat_tau: float = 1.5
    mirostat_eta: float = 0.3
    add_bos_token: bool = True
    ban_eos_token: bool = False
    skip_special_tokens: bool = True
    logit_bias: dict[str, float] | None = None
    negative_prompt: str | None = None
    json_schema: dict | None = None
    regex_pattern: str | None = None
    grammar_string: str | None = None
    speculative_ngram: bool = True
    typical: float = 1.0
    penalty_range: float = 0.0
    cfg_scale: float = 1.0
    max_temp: float = 1.0
    min_temp: float = 1.0
    temp_exponent: float = 1.0
    banned_tokens: list[int] | None = None


class Message(TypedDict):
    role: str
    content: str


class TabbyApiList(TypedDict):
    object: typing.Literal["list"]
    data: list[T]


class ModelCardParameters(TypedDict, total=False):
    max_seq_len: int
    rope_scale: int
    rope_alpha: int
    cache_size: int
    cache_mode: str
    prompt_template: str
    num_experts_per_token: int
    draft: "ModelCard"


class ModelCard(TypedDict, total=False):
    id: str
    object: str = "Model"
    created: int
    owned_by: str
    # logging:
    parameters: ModelCardParameters


class Tabby:
    """Talk to TabbyAPI running an LLM over HTTP."""

    def __init__(self, api_root: str, auth_key: str, admin_key: str | None):
        self.api_root = api_root
        self.__api_key = auth_key
        self.__admin_key = admin_key

    @classmethod
    def from_app_config(cls, appname) -> "Tabby":
        # Linux
        home = Path(os.environ["HOME"])
        xdg_config = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config"))
        config_filename = xdg_config / appname / "config.ini"
        # TODO: OSX, Windows

        with open(config_filename) as fh:
            cfg = configparser.ConfigParser()
            cfg.read_string("[DEFAULT]\n" + fh.read())
            config = cfg["DEFAULT"]

        api_root = config["api_root"]
        if api_root and not api_root.endswith("/"):
            api_root += "/"
        if not api_root:
            api_root = "http://localhost:5000/"
        auth_key = config["api_key"]
        admin_key = config.get("admin_key")
        return cls(api_root, auth_key, admin_key)

    def set_auth_headers(self, headers) -> None:
        headers["x-api-key"] = self.__api_key
        if self.__admin_key:
            headers["x-admin-key"] = self.__admin_key

    def get(self, path: str, **kw: str) -> requests.Response:
        headers = kw.get("headers") or {}
        self.set_auth_headers(headers)
        kw["headers"] = headers
        url = self.api_root + path
        try:
            response = requests.get(url, **kw)
            return response
        except requests.exceptions.RequestException as e:
            if hasattr(e.response, "text"):
                raise Exception(
                    f"Error making GET request to {url}: {str(e)} - {e.response.text}"
                )
            raise Exception(f"Error making GET request to {url}: {str(e)}")

    def post(self, path: str, data: str, **kw: str) -> requests.Response:
        headers = kw.get("headers") or {}
        self.set_auth_headers(headers)
        kw["headers"] = headers
        url = self.api_root + path
        try:
            response = requests.post(url, data=data, **kw)
            return response
        except requests.exceptions.RequestException as e:
            if hasattr(e.response, "text"):
                raise Exception(
                    f"Error making POST request to {url}: {str(e)} - {e.response.text}"
                )
            raise Exception(f"Error making POST request to {url}: {str(e)}")

    def get_loaded_model(self) -> ModelCard:
        # Get loaded model
        # Send with api key: X-Api-Key header (or bearer auth)
        return self.get("v1/model").json()

    def get_available_models(self) -> list[ModelCard]:
        # Get available models
        # Send with api key: X-Api-Key header (or bearer auth)
        resp: TabbyApiList[ModelCard] = self.get("v1/model/list").json()
        return resp["data"]

    def get_available_draft_models(self) -> list[ModelCard]:
        # Get available draft models
        # Send with api key: X-Api-Key header (or bearer auth)
        resp: TabbyApiList[ModelCard] = self.get("v1/model/draft/list").json()
        return resp["data"]

    def unload(self) -> None:
        self.post("v1/model/unload", None)

    def swap_model(self, new_model: str, config: ModelConfig) -> None:
        # Select model
        # Send with admin key: X-Admin-Key header (or bearer auth)
        self.unload()

        data: dict[str, float | int | str | dict] = {
            "name": new_model,
        }
        if config.cache_mode:
            data["cache_mode"] = config.cache_mode

        if config.prompt_template:
            data["prompt_template"] = config.prompt_template

        if config.max_seq_len:
            data["max_seq_len"] = config.max_seq_len

        if hasattr(config, 'draft') and config.draft:
            data["draft"] = config.draft

        postdata = json.dumps(data)

        return self.post("v1/model/load", data=postdata, stream=True)

    def completions(self, prompt: str, settings: SamplerSettings = SamplerSettings()):
        """
        Generate text completion from a prompt.
        """
        data = {"prompt": prompt, **settings}
        return self.post("v1/completions", json.dumps(data)).json()

    def chat_completions(
        self, messages: list[Message], settings: SamplerSettings = SamplerSettings()
    ):
        """
        Generate the next message in a chat.
        """
        data = {
            "messages": [message for message in messages],
            **settings,
        }
        return self.post("v1/chat/completions", json.dumps(data)).json()

    def encode_tokens(self, text: str) -> EncodeTokensResponse:
        data = {
            "text": text,
            "add_bos_token": True,
            "encode_special_tokens": True,
            "decode_special_tokens": True,
        }
        return self.post("v1/token/encode", json.dumps(data)).json()

    def model_names(self):
        models = self.get_available_models()
        models.sort(key=lambda x: x["id"])
        return [m["id"] for m in models]


def simple_term_similarity(doc_term: str, query_term: str) -> float:
    scale = 1.0
    if query_term.isdigit():
        scale = 0.25

    if doc_term == query_term:
        return 2.0 * scale
    elif doc_term.startswith(query_term):
        return len(query_term) / len(doc_term) * 0.5 * scale
    elif doc_term.endswith(query_term):
        return len(query_term) / len(doc_term) * 0.45 * scale
    elif query_term in doc_term:
        return len(query_term) / len(doc_term) * 0.05 * scale
    return 0.0


def extract_query_term_groups(query: list[str]) -> list[list[str]]:
    return [
        [term.lower()] + [w.lower() for w in re.findall(LU_WORD, p)]
        for term in query
        for p in re.split(r"[-_\s]", term)
    ]


def extract_doc_term_groups(doc: str) -> list[list[str]]:
    out = []
    for p in re.split("[-_ ]", doc):
        doc_term_group = []
        out.append(doc_term_group)
        subword = re.findall(LU_WORD, p)
        doc_term_group.append(p.lower())
        if len(subword) == 1 and doc_term_group[-1:] != [p.lower()]:
            doc_term_group.append(p.lower())
        else:
            for w in subword:
                if doc_term_group[-1:] != [w.lower()]:
                    doc_term_group.append(w.lower())
    return out


def fuzzy_match(
    corpus: list[str],
    query: list[str],
    bias_fn: typing.Callable[[list[str]], float] | None = None,
) -> list[tuple[str, float]]:

    # First, check for an exact selection.
    for doc in corpus:
        if [doc] == query:
            return [(doc, float("Inf"))]

    query_parts: list[list[str]] = extract_query_term_groups(query)

    # Index the models by name parts: dash, underscore, lower to upper, lower
    # to digit transitions all count as word separators, but group together
    # unapunctuated text so that only the best matches within the group count.
    doc_term_corpus: dict[str, list[list[str]]] = {}
    for doc in corpus:
        doc_term_corpus[doc] = extract_doc_term_groups(doc)
        # print(doc, "->", doc_term_corpus[doc])

    # Find the documents where the terms have the most overlap with the query.
    modelscore: dict[str, float] = defaultdict(float)
    partial_matches = []
    for doc, doc_term_groups in doc_term_corpus.items():
        term_group_hits = set()
        for doc_term_group in doc_term_groups:
            for tgid, term_group in enumerate(query_parts):

                # For each document term group find the best match for the
                # query term group and disregard lesser matches.
                groupscore = {t: 0.0 for t in doc_term_group}
                tscore = 0.0
                for query_term in term_group:
                    for doc_term in doc_term_group:
                        similarity = simple_term_similarity(doc_term, query_term)
                        groupscore[doc_term] += similarity
                        # print( doc_term, ' x ', query_term, '->', similarity)
                tscore = max(groupscore.values()) / len(query_term)
                if tscore > 0:
                    term_group_hits.add(tgid)
                    modelscore[doc] += tscore

        # Only accept documents that had matches for all query terms.
        if len(term_group_hits) < len(query_parts):
            modelscore[doc] = 0
            if term_group_hits:
                partial_matches.append(doc)
        else:
            # Apply the caller's finger on the scales on a good hit.
            if modelscore[doc] > 0 and bias_fn:
                modelscore[doc] += bias_fn(doc_term_groups)

    n = list(modelscore.items())
    n.sort(key=lambda x: x[1], reverse=True)
    top_hits = [(name, score) for name, score in n if score > 0]
    return top_hits


def display_progress(r: requests.Response):
    err = None
    with tqdm(unit=" Layers") as pb:
        pb.refresh()
        for line in r.iter_lines(chunk_size=None):
            if not line:
                continue
            line = line.removeprefix(b"data: ")
            j = json.loads(line)
            if "modules" in j:
                n = int(j.get("module", 0))
                tot = int(j.get("modules", 1))
                pb.total = tot
                pb.n = n
                pb.refresh()
            else:
                # allocate a line above the progress bar and fill it with the full error
                pb.clear()
                print()
                pb.display(line.decode("utf-8"), pos=-1)

                # Could do this above, but there's more info than just the error message sometimes.
                if "error" in j:
                    err = j["error"]["message"]
                    break

    # display the error after the progress bar
    # cleans up its UI.
    if err:
        print(err)
    return not err


def exllama2_bias(parts: list[str]) -> float:
    bias = 0
    for group in parts:
        match group:
            case ["exl2", "exl", "2"]:
                bias += 1.0
            case ["gptq"]:
                bias += 0.25
    return bias


def fuzzy_select_model(tabby: Tabby, query: list[str], args, is_draft: bool = False) -> bool:
    model_names = tabby.get_available_draft_models() if is_draft else tabby.model_names()
    model_names = [m["id"] for m in model_names]
    models = fuzzy_match(model_names, query, bias_fn=exllama2_bias)
    # select the models with the top score
    models = [m for m, s in models if s == models[0][1]]
    if len(models) == 0:
        print(f"No matching {'draft ' if is_draft else ''}models")
        return False
    if len(models) != 1:
        print(f"Multiple matching {'draft ' if is_draft else ''}models:")
        for model in models:
            print(model)
        return False
    else:
        new_model = models[0]
        if is_draft:
            args.draft = {"name": new_model}
        r = tabby.swap_model(new_model, args)
        return display_progress(r)

def scaled_int(value, base=10):
    scale = 1
    if isinstance(value, str):
        trunc = 0
        if value.endswith('k'):
            scale = 1000
            trunc = -1
        elif value.endswith('ki'):
            scale = 1024
            trunc = -2
        elif value.endswith('m') or value.endswith('M'):
            scale = 1000_000
            trunc = -1
        elif value.endswith('mi') or value.endswith('Mi'):
            scale = 1024*1024
            trunc = -2
        if trunc:
            value = value[:trunc]
    return int(value, base) * scale


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mode", metavar="mode", type=str, nargs="?")
    argparser.add_argument(
        "--ctx",
        "--context-length",
        "-C",
        type=scaled_int,
        dest="max_seq_len",
        required=False,
    )
    argparser.add_argument(
        "-8",
        "-8k",
        action='store_const',
        const=8192,
        dest="max_seq_len",
        required=False,
    )
    argparser.add_argument(
        "-16",
        "-16k",
        action='store_const',
        const=16834,
        dest="max_seq_len",
        required=False,
    )
    argparser.add_argument(
        "-32",
        "-32k",
        action='store_const',
        const=32768,
        dest="max_seq_len",
        required=False,
    )
    argparser.add_argument(
        "-48",
        "-48k",
        action='store_const',
        const=43008,
        dest="max_seq_len",
        required=False,
    )
    argparser.add_argument(
        "-128",
        "-128k",
        action='store_const',
        const=131072,
        dest="max_seq_len",
        required=False,
    )
    argparser.add_argument(
        "-T",
        "--prompt-template",
        type=str,
        dest="prompt_template",
        default="",
        required=False,
    )
    argparser.add_argument(
        "--cache-mode",
        type=str,
        dest="cache_mode",
        default="Q4",
        choices=["Q4", "FP8", "FP16"],
        required=False,
    )
    argparser.add_argument("words", metavar="modelword", type=str, nargs="*")
    argparser.add_argument(
        "--draft-model", "-D", type=str, dest="draft_model", required=False,
        help="Select a draft model to use alongside the main model"
    )
    args = argparse.Namespace()
    args, unknown = argparser.parse_known_args(namespace=args)
    tabby = Tabby.from_app_config("tabswitch")

    match args.mode:
        case "help":
            print(
                "current, list, set-model <exact name>, "
                "model <approximate name>, select-model <approximate name>, "
                "or just <approximate name>"
            )

        case None | "current":
            print(tabby.get_loaded_model())

        case "unload":
            tabby.unload()

        case "ls" | "list" | "list-models":
            query = args.words
            if query:
                model_names = tabby.model_names()
                models = fuzzy_match(model_names, query, bias_fn=exllama2_bias)
                for m, score in models:
                    print(f"{score:0.3}\t{m}")
            else:
                models = tabby.get_available_models()
                models = [m["id"] for m in models]
                models.sort()
                for m in models:
                    print(m)

        case "set-model":
            new_model = args.words[0]
            r = tabby.swap_model(new_model, args)
            display_progress(r) or sys.exit(1)

        case "model" | "select-model":
            new_model = args.words
            if not fuzzy_select_model(tabby, new_model, args):
                sys.exit(1)
            if args.draft_model:
                if not fuzzy_select_model(tabby, [args.draft_model], args, is_draft=True):
                    sys.exit(1)

        case _:
            new_model = [args.mode] + args.words
            if not fuzzy_select_model(tabby, new_model, args):
                sys.exit(1)
            if args.draft_model:
                if not fuzzy_select_model(tabby, [args.draft_model], args, is_draft=True):
                    sys.exit(1)


if __name__ == "__main__":
    main()

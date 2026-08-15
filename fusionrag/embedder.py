import logging
import os
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from huggingface_hub import hf_hub_download, list_repo_files  # noqa: E402

REPO = "Xenova/all-MiniLM-L6-v2"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

ONNX_CANDIDATES = [
    "onnx/model.onnx",
    "onnx/encoder_model.onnx",
    "model.onnx",
]


def download(repo=REPO, dest=MODELS_DIR):
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    dest = Path(dest) / repo
    dest.mkdir(parents=True, exist_ok=True)

    files = list_repo_files(repo_id=repo)
    onnx_file = next((c for c in ONNX_CANDIDATES if c in files), None)
    if not onnx_file:
        raise FileNotFoundError(f"No ONNX model found in {repo}")

    wanted = [("tokenizer.json", "tokenizer.json"), (onnx_file, "model.onnx")]
    if onnx_file + "_data" in files:
        wanted.append((onnx_file + "_data", "model.onnx_data"))

    for remote, local in wanted:
        dst = dest / local
        if not dst.exists():
            src = hf_hub_download(repo_id=repo, filename=remote)
            shutil.copy2(src, dst)
            print(f"  saved {dst}")
    return dest


class Embedder:
    def __init__(self, path=None):
        path = Path(path) if path else MODELS_DIR / REPO
        if not (path / "model.onnx").exists():
            path = download()
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        # configure once: mutating the tokenizer per call is not thread-safe
        # ("Already borrowed" under concurrent encodes)
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_length=512)
        self.session = ort.InferenceSession(
            str(path / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    def encode(self, text, normalize=True):
        return self.encode_batch([text], normalize=normalize)[0]

    def encode_batch(self, texts, normalize=True):
        encoded = self.tokenizer.encode_batch(texts)
        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = np.array([e.ids for e in encoded], dtype=np.int64)
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )
        hidden = self.session.run(None, feed)[0]
        mask = feed["attention_mask"][..., None]
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        if normalize:
            pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled

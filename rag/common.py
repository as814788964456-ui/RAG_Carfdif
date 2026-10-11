"""RAG 共用設定：路徑、embedding 模型、索引讀取。"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "data" / "knowledge-base"
GOLDEN_PATH = KB_DIR / "test-set" / "golden_set.jsonl"
INDEX_DIR = ROOT / "data" / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"

EMBEDDER_KEYS = ["bge-m3", "xcloud", "xcloud-8k"]


def get_embedder(key):
    if key == "bge-m3":
        from embedding.bge_m3 import BgeM3Embedder

        return BgeM3Embedder()
    from embedding.xcloud import XCloudEmbedder

    return XCloudEmbedder(variant="8k" if key == "xcloud-8k" else "standard")


def vectors_path(key):
    return INDEX_DIR / f"{key}.npy"


def load_chunks():
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_vectors(key):
    return np.load(vectors_path(key))

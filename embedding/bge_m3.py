"""BAAI/bge-m3 embedding（1024 維，dense 向量）。

    embed_queries(list[str]) -> np.ndarray  (N, 1024)，已 L2 正規化
    embed_documents(list[str]) -> np.ndarray (N, 1024)，已 L2 正規化
所以相似度直接用內積（= cosine）即可。

用法:
    python bge_m3.py "文字一" "文字二" ...   # 印出兩兩相似度
"""
import numpy as np


class BgeM3Embedder:
    """查詢與文件不需加前綴。"""

    name = "bge-m3"
    repo_id = "BAAI/bge-m3"

    def __init__(self, device=None, max_length=8192):
        import torch
        from sentence_transformers import SentenceTransformer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(
            self.repo_id,
            device=device,
            model_kwargs={"dtype": torch.float16 if device == "cuda" else torch.float32},
        )
        self.model.max_seq_length = max_length

    def _embed(self, texts, batch_size=16):
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

    def embed_queries(self, texts):
        return self._embed(texts)

    def embed_documents(self, texts):
        return self._embed(texts)


if __name__ == "__main__":
    import sys

    texts = sys.argv[1:] or ["台灣最高的山有多高？", "玉山主峰海拔 3,952 公尺。", "珍珠奶茶起源於台中。"]
    vecs = BgeM3Embedder().embed_documents(texts)
    print(f"維度 {vecs.shape[1]}")
    print(np.round(vecs @ vecs.T, 3))

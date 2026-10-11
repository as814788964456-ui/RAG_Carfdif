"""xVITA-Embed-300M-zhTW embedding（GGUF 檔名 xcloud-emb-zhtw，768 維）。

    embed_queries(list[str]) -> np.ndarray  (N, 768)，已 L2 正規化
    embed_documents(list[str]) -> np.ndarray (N, 768)，已 L2 正規化
所以相似度直接用內積（= cosine）即可。

用法:
    python xcloud.py "文字一" "文字二" ...   # 印出兩兩相似度
"""
import numpy as np


class XCloudEmbedder:
    """以 llama.cpp 執行，需使用 EmbeddingGemma 格式的前綴:
        查詢: "task: search result | query: ..."
        文件: "title: none | text: ..."
    variant="8k" 為長文特化版（整篇文件檢索），預設為標準版（段落級 RAG）。
    """

    name = "xcloud-emb-zhtw"
    repo_id = "xCloudinfo/xVITA-Embed-300M-zhTW-GGUF"
    QUERY_PREFIX = "task: search result | query: "
    DOC_PREFIX = "title: none | text: "

    def __init__(self, variant="standard", quant="Q8_0", n_ctx=8192, n_gpu_layers=0):
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        suffix = "-8k" if variant == "8k" else ""
        filename = f"xcloud-emb-zhtw{suffix}-{quant}.gguf"
        self.name = f"xcloud-emb-zhtw{suffix}"
        path = hf_hub_download(self.repo_id, filename)
        # 非因果 embedding 模型需整段放進同一個 ubatch，所以 n_batch = n_ubatch = n_ctx
        self.model = Llama(
            model_path=path,
            embedding=True,
            n_ctx=n_ctx,
            n_batch=n_ctx,
            n_ubatch=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def _embed(self, texts):
        # 逐筆送入：一次傳整個 list 會把多段文字打包進同一個 ubatch，注意力運算量隨總長度平方成長，反而慢一倍以上
        vecs = np.asarray([self.model.embed(t, truncate=True) for t in texts], dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def embed_queries(self, texts):
        return self._embed([self.QUERY_PREFIX + t for t in texts])

    def embed_documents(self, texts):
        return self._embed([self.DOC_PREFIX + t for t in texts])


if __name__ == "__main__":
    import sys

    texts = sys.argv[1:] or ["台灣最高的山有多高？", "玉山主峰海拔 3,952 公尺。", "珍珠奶茶起源於台中。"]
    vecs = XCloudEmbedder().embed_documents(texts)
    print(f"維度 {vecs.shape[1]}")
    print(np.round(vecs @ vecs.T, 3))

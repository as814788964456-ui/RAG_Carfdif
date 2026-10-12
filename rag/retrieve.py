"""檢索：語意（embedding）、關鍵字（BM25），以及多路結果以加權 RRF 融合。

檢索設定用「+」組合，BM25 可用「*權重」調整在融合中的比重:
    bge-m3                  只用 bge-m3
    bge-m3+xcloud           兩個 embedding 模型融合（預設，評估結果最好）
    xcloud+bm25*0.3         xcloud 加上權重 0.3 的 BM25

用法（在 TAIDE 根目錄執行，需先跑 python -m rag.ingest）:
    python -m rag.retrieve "忘記密碼怎麼辦"
    python -m rag.retrieve -c bm25 "LIFE-Core 登入失敗"
程式中:
    from rag.retrieve import Retriever
    r = Retriever("bge-m3+xcloud")
    for chunk, score in r.search("忘記密碼怎麼辦", k=5): ...
"""
import math
import re
from collections import Counter

import numpy as np

from rag.common import EMBEDDER_KEYS, get_embedder, load_chunks, load_vectors

DEFAULT_CONFIG = "bge-m3+xcloud"
RRF_K = 60  # RRF 常數，越大則排名靠後的結果影響越大
ASCII_WORD = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
CJK_RUN = re.compile(r"[㐀-鿿]+")
# 口語問句的虛詞。文件是公文語氣、很少出現這些字，BM25 會誤判為稀有關鍵字而壓過真正的關鍵字，所以查詢時先移除
QUERY_FILLERS = re.compile(
    "|".join(sorted([
        "是什麼", "什麼是", "什麼", "怎麼辦", "要怎麼", "怎麼", "怎樣", "如何", "請問", "可以", "可不可以",
        "需要", "要不要", "會不會", "有沒有", "是不是", "哪裡", "哪些", "多久", "幾天", "我的", "我們", "我",
        "嗎", "呢", "啊", "吧", "了", "的",
    ], key=len, reverse=True))
)


def tokenize(text):
    """英數字取整個詞（LIFE-Core、P1、vip），中文取相鄰兩字（bigram）。不需斷詞詞典，錯字只影響兩個 bigram。"""
    text = text.lower()
    tokens = ASCII_WORD.findall(text)
    for run in CJK_RUN.findall(text):
        tokens += [run] if len(run) == 1 else [run[i:i + 2] for i in range(len(run) - 1)]
    return tokens


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg_len = self.lens.mean()
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query_tokens):
        out = np.zeros(len(self.tfs), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.lens / self.avg_len)
        for t in set(query_tokens):
            if t not in self.idf:
                continue
            tf = np.array([d.get(t, 0) for d in self.tfs], dtype=np.float32)
            out += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
        return out


def parse_config(config):
    """「bge-m3+bm25*0.3」→ [("bge-m3", 1.0), ("bm25", 0.3)]"""
    parts = []
    for part in config.split("+"):
        name, _, weight = part.partition("*")
        if name not in EMBEDDER_KEYS + ["bm25"]:
            raise ValueError(f"未知的檢索來源：{name}（可用 {EMBEDDER_KEYS + ['bm25']}）")
        parts.append((name, float(weight or 1)))
    return parts


def weighted_rrf(parts, mask):
    """各路分數各自排名後以 權重/(RRF_K + 名次) 加總；分數 ≤ 0（如 BM25 完全沒命中）的不計分。"""
    fused = np.zeros(len(mask), dtype=np.float32)
    for scores, weight in parts:
        order = np.argsort(-np.where(mask, scores, -np.inf))
        ranks = np.empty(len(order), dtype=np.float32)
        ranks[order] = np.arange(1, len(order) + 1)
        fused += np.where(mask & (scores > 0), weight / (RRF_K + ranks), 0)
    return fused


class Retriever:
    def __init__(self, config=DEFAULT_CONFIG, include_deprecated=False, embedders=None):
        """embedders: 可傳入 {模型名稱: embedder} 重複使用已載入的模型，缺少的會自動載入並放回此 dict。"""
        self.config = config
        self.parts = parse_config(config)
        self.chunks = load_chunks()
        self.mask = np.array([include_deprecated or c["status"] != "已廢止" for c in self.chunks])
        embedders = {} if embedders is None else embedders
        self.dense = {}
        for name, _ in self.parts:
            if name == "bm25":
                self.bm25 = BM25([tokenize(c["content"]) for c in self.chunks])
            else:
                if name not in embedders:
                    embedders[name] = get_embedder(name)
                self.dense[name] = (embedders[name], load_vectors(name))

    def scores(self, query):
        """回傳每個 chunk 的分數（越高越相關），被排除的 chunk 為 -inf。
        單一來源時為原始分數（cosine 或 BM25），多來源時為 RRF 分數。
        query 可以是多個查詢的 list（例如多輪對話中「本輪問題」與「前一輪＋本輪」），各自檢索後以 RRF 融合。"""
        if isinstance(query, (list, tuple)):
            if len(query) == 1:
                return self.scores(query[0])
            fused = weighted_rrf([(self.scores(q), 1.0) for q in query], self.mask)
            return np.where(self.mask, fused, -np.inf)
        parts = []
        for name, weight in self.parts:
            if name == "bm25":
                s = self.bm25.scores(tokenize(QUERY_FILLERS.sub(" ", query)))
            else:
                embedder, vectors = self.dense[name]
                s = vectors @ embedder.embed_queries([query])[0]
            parts.append((s, weight))
        scores = parts[0][0] if len(parts) == 1 else weighted_rrf(parts, self.mask)
        return np.where(self.mask, scores, -np.inf)

    def search(self, query, k=5):
        scores = self.scores(query)
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG, help="檢索設定，如 bge-m3、bge-m3+xcloud、xcloud+bm25*0.3")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    for rank, (chunk, score) in enumerate(Retriever(args.config).search(args.query, args.k), 1):
        print(f"{rank}. ({score:.4f}) {chunk['doc_id']}#{chunk['section']}｜{chunk['text'][:50]}…")

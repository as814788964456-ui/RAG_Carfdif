"""語意檢索示範：用 bge-m3 與 xcloud-emb-zhtw 對同一批文件做檢索並比較結果。

用法:
    python search_demo.py                        # 跑內建的範例問題，兩個模型都用
    python search_demo.py "要怎麼報稅？"          # 自訂問題
    python search_demo.py -m xcloud "要怎麼報稅？" # 只用其中一個模型
    python search_demo.py --docs my_docs.json    # 換成自己的文件（JSON 字串陣列）
"""
import argparse
import json
import time
from pathlib import Path

from bge_m3 import BgeM3Embedder
from xcloud import XCloudEmbedder

HERE = Path(__file__).resolve().parent
EMBEDDERS = {
    "bge-m3": BgeM3Embedder,
    "xcloud": XCloudEmbedder,
    "xcloud-8k": lambda: XCloudEmbedder(variant="8k"),
}
DEFAULT_QUERIES = [
    "台灣最高的山有多高？",
    "第一次辦護照要帶什麼？",
    "什麼時候要繳所得稅？",
    "老闆叫我加班可以不給錢嗎？",
    "颱風天會不會放假要看哪裡？",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", nargs="*", help="要檢索的問題，未提供則使用內建範例")
    parser.add_argument("-m", "--models", nargs="+", choices=EMBEDDERS, default=["bge-m3", "xcloud"])
    parser.add_argument("--docs", type=Path, default=HERE / "sample_docs.json")
    parser.add_argument("-k", "--top-k", type=int, default=3)
    args = parser.parse_args()

    docs = json.loads(args.docs.read_text(encoding="utf-8"))
    queries = args.queries or DEFAULT_QUERIES

    results = {}
    for key in args.models:
        t0 = time.perf_counter()
        embedder = EMBEDDERS[key]()
        doc_vecs = embedder.embed_documents(docs)
        query_vecs = embedder.embed_queries(queries)
        scores = query_vecs @ doc_vecs.T
        results[embedder.name] = scores
        print(f"[{embedder.name}] 維度 {doc_vecs.shape[1]}，載入+編碼 {time.perf_counter() - t0:.1f} 秒")

    for qi, query in enumerate(queries):
        print(f"\n問題：{query}")
        for name, scores in results.items():
            print(f"  ── {name}")
            for rank, di in enumerate(scores[qi].argsort()[::-1][: args.top_k], 1):
                print(f"    {rank}. ({scores[qi, di]:.3f}) {docs[di][:40]}…")


if __name__ == "__main__":
    main()

"""檢索評估：用 golden_set 的問題（原句＋口語／錯字變體）計算 Hit@k 與 MRR。

一個 chunk 算命中：doc_id 相同，且章節路徑中有標題對應到 golden set 的 `doc_id#章節`
（例如 `IT-FAQ-001#Q2` 對應標題「Q2. 忘記密碼怎麼辦？」）。

用法（在 TAIDE 根目錄執行，需先跑 python -m rag.ingest）:
    python -m rag.eval                                  # 預設：各模型單獨、bm25、所有模型融合
    python -m rag.eval -c bge-m3+xcloud "xcloud+bm25*0.3"  # 指定檢索設定（寫法見 rag/retrieve.py）
    python -m rag.eval --keep-deprecated                # 不排除已廢止文件，觀察版本衝突
輸出:
    終端機印出指標表；每題的檢索結果存到 data/eval/retrieval_<設定>.csv
"""
import argparse
import csv
import json
import unicodedata

import numpy as np

from rag.common import EMBEDDER_KEYS, GOLDEN_PATH, ROOT, load_chunks, vectors_path
from rag.retrieve import Retriever

EVAL_DIR = ROOT / "data" / "eval"
TOP_K = 10


def heading_matches(heading, target):
    """「Q2」對應「Q2. 忘記密碼怎麼辦？」，但不對應「Q20…」。"""
    if heading == target:
        return True
    return heading.startswith(target) and not heading[len(target)].isalnum()


def chunk_matches(chunk, source):
    doc_id, _, title = source.partition("#")
    return chunk["doc_id"] == doc_id and any(heading_matches(h, title) for h in chunk["path"])


def load_queries():
    """每題展開成原句與變體，每筆帶 golden set 的欄位。"""
    queries = []
    with GOLDEN_PATH.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            for form, text in [("原句", item["question"])] + [("變體", v) for v in item["variants"]]:
                queries.append({**item, "form": form, "query": text})
    return queries


def evaluate(retriever, chunks, queries):
    deprecated = np.array([c["status"] == "已廢止" for c in chunks])
    records = []
    for q in queries:
        row = retriever.scores(q["query"])
        top = np.argsort(-row)[:TOP_K]
        rec = {"q": q, "top": top, "top1_score": float(row[top[0]])}
        if q["answerable"]:
            relevant = [i for i, c in enumerate(chunks) if any(chunk_matches(c, s) for s in q["sources"])]
            ranks = [r for r, i in enumerate(top, 1) if i in relevant]
            rec["rank"] = ranks[0] if ranks else None
            covered = {s for s in q["sources"] if any(chunk_matches(chunks[i], s) for i in top[:5])}
            rec["recall5"] = len(covered) / len(q["sources"])
        rec["deprecated_top3"] = bool(deprecated[top[:3]].any())
        records.append(rec)
    return records


def metrics(records):
    n = len(records)
    ranks = [r["rank"] for r in records]
    hit = lambda k: sum(1 for x in ranks if x and x <= k) / n
    mrr = sum(1 / x for x in ranks if x) / n
    return {"n": n, "Hit@1": hit(1), "Hit@3": hit(3), "Hit@5": hit(5), "MRR": mrr,
            "Recall@5": sum(r["recall5"] for r in records) / n}


def pad(text, width):
    """中文字佔兩格，補空白對齊。"""
    w = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(width - w, 1)


def print_table(title, results, group_fn):
    cols = ["Hit@1", "Hit@3", "Hit@5", "MRR", "Recall@5"]
    print(f"\n## {title}")
    print(pad("分組", 18) + pad("設定", 16) + f"{'題數':>4}" + "".join(f"{c:>10}" for c in cols))
    groups = sorted({group_fn(r["q"]) for recs in results.values() for r in recs})
    for g in groups:
        for key, recs in results.items():
            sub = [r for r in recs if group_fn(r["q"]) == g]
            m = metrics(sub)
            print(pad(g, 18) + pad(key, 16) + f"{m['n']:>6}" + "".join(f"{m[c]:>10.3f}" for c in cols))


def save_csv(key, records, chunks):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / f"retrieval_{key.replace('*', 'x')}.csv"
    label = lambda i: f"{chunks[i]['doc_id']}#{chunks[i]['section']}"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question_type", "form", "query", "sources", "rank", "top1_score", "top1", "top2", "top3"])
        for r in records:
            q = r["q"]
            w.writerow([q["id"], q["question_type"], q["form"], q["query"], "；".join(q["sources"]),
                        r.get("rank") or "", f"{r['top1_score']:.3f}", *[label(i) for i in r["top"][:3]]])
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configs", nargs="+", help="檢索設定，如 bge-m3、bge-m3+xcloud、xcloud+bm25*0.3")
    parser.add_argument("--keep-deprecated", action="store_true", help="不排除已廢止文件")
    args = parser.parse_args()
    models = [k for k in EMBEDDER_KEYS if vectors_path(k).exists()]
    if not models:
        raise SystemExit("找不到索引，請先執行 python -m rag.ingest")
    configs = args.configs or models + ["bm25"] + (["+".join(models)] if len(models) > 1 else [])

    chunks = load_chunks()
    queries = load_queries()
    for q in queries:
        for s in q["sources"]:
            if not any(chunk_matches(c, s) for c in chunks):
                print(f"警告：{q['id']} 的 source「{s}」找不到對應的 chunk")

    embedders = {}  # 各設定共用已載入的模型
    all_records = {
        config: evaluate(Retriever(config, args.keep_deprecated, embedders), chunks, queries) for config in configs
    }
    answerable = {k: [r for r in recs if r["q"]["answerable"]] for k, recs in all_records.items()}

    print(f"\n可回答題 {len(answerable[next(iter(answerable))]) // 3} 題 × 3 種問法；已廢止文件：{'保留' if args.keep_deprecated else '排除'}")
    print_table("整體", answerable, lambda q: "全部")
    print_table("依問法", answerable, lambda q: q["form"])
    print_table("依題型", answerable, lambda q: q["question_type"])
    print_table("依類別", answerable, lambda q: q["category"])

    print("\n## 拒答門檻參考（純語意的 top-1 相似度）")
    for key, recs in all_records.items():
        if key not in EMBEDDER_KEYS:
            continue
        ans = np.array([r["top1_score"] for r in recs if r["q"]["answerable"]])
        una = np.array([r["top1_score"] for r in recs if not r["q"]["answerable"]])
        print(f"{key:<14}可回答題 中位數 {np.median(ans):.3f}、最低 10% {np.percentile(ans, 10):.3f}"
              f"｜無答案題 中位數 {np.median(una):.3f}、最高 {una.max():.3f}")

    if args.keep_deprecated:
        print("\n## 已廢止文件出現在 top-3 的次數")
        for key, recs in all_records.items():
            print(f"{key:<14}{sum(r['deprecated_top3'] for r in recs)} / {len(recs)}")

    print()
    for key, recs in all_records.items():
        print(f"[{key}] 每題結果 → {save_csv(key, recs, chunks).relative_to(ROOT)}")


if __name__ == "__main__":
    main()

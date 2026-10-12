"""用 TAIDE 對黃金測試集生成答案，並自動評分。

讀 data/contexts/golden.jsonl（python -m rag.context --golden 匯出，內含檢索結果），
依指定的 prompt 版本重組 messages 後生成，所以生成時不需要載入 embedding 模型。

用法（在 TAIDE 根目錄執行）:
    python -m rag.answer -p v2                      # 68 題全部生成並評分 → data/answers/llama_v2.jsonl
    python -m rag.answer -p v2 --ids IT-01 UNA-01   # 只跑指定題目
    python -m rag.answer --score llama_v1 llama_v2  # 只評分，比較多次執行的結果
中斷後重跑同一個指令會從還沒做的題目繼續。

自動評分項目（不需要人工或其他 LLM）:
    有標出處      回答中有沒有 [文件編號#章節]（一個方括號內可有多個，以逗號分隔）
    出處有效率    引用的段落是否真的在這題的檢索結果裡；章節名稱被改寫但編號相同（Q2、第三條、四、）也算
    引用正確段落  是否引用到黃金集標示的正確段落
    關鍵事實命中  標準答案中的數字、網址、代碼（分機、表單編號等）有幾成出現在回答中
    誤拒          可回答題只回答了標準拒答句（≤ 80 字）
    表示查不到    回答中出現「查不到」「並無」「未提及」等說法（不論後面是否又加了推測）
    簡潔拒答      使用標準拒答句且全文 ≤ 80 字，沒有附帶推測或額外建議
    引用 EX-001   照抄了 v2 prompt 範例中的虛構文件
    引用廢止      回答中出現已廢止的舊版 VPN 文件內容
"""
import argparse
import csv
import json
import re
import time

from rag.common import ROOT, load_chunks
from rag.context import CONTEXT_DIR, DEFAULT_PROMPT, PROMPTS, REFUSAL, build_user_message
from rag.eval import heading_matches, pad

ANSWER_DIR = ROOT / "data" / "answers"
BRACKET = re.compile(r"\[([^\[\]]*#[^\[\]]*)\]")  # 方括號內含 # 的就是引用，可能一次放多個（以逗號分隔）
CITATION = re.compile(r"^\s*([A-Z]+(?:-[A-Z]+)*-\d{3}(?:-OLD)?)#(.+?)\s*$")
# 章節編號：Q2、第三條、第七章、四、、3.2、單元四。模型常改寫章節名稱但保留編號，以編號比對
SECTION_ID = re.compile(r"^(Q\d+|第[一二三四五六七八九十百]+[條章節]|[一二三四五六七八九十]+、|\d+(?:\.\d+)*|單元[一二三四五六七八九十]+)")
FACT = re.compile(r"https?://[\w./-]+|[A-Z]{1,5}-?\d+|\d[\d,]*(?:\.\d+)?")
DEPRECATED_MARKERS = ["IT-FAQ-002-OLD", "AuroraLink", "aurora-old"]
# 表示資料不足但沒有用標準拒答句的說法
DECLINE_MARKERS = [REFUSAL, "查不到", "無法找到", "無法提供", "並無", "未明確", "未提及", "沒有提到"]


def parse_citations(answer):
    out = []
    for inside in BRACKET.findall(answer):
        for part in re.split(r"[,，;；]", inside):
            m = CITATION.match(part)
            if m:
                out.append((m.group(1), m.group(2)))
    return out


def same_section(a, b):
    """章節名稱相同、互為前綴，或開頭的章節編號相同（Q2 與「Q2. 忘記密碼怎麼處理」）。"""
    if a == b or heading_matches(a, b) or heading_matches(b, a):
        return True
    ma, mb = SECTION_ID.match(a), SECTION_ID.match(b)
    return bool(ma and mb and ma.group(1) == mb.group(1))


def load_items(path=CONTEXT_DIR / "golden.jsonl"):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_messages(item, prompt, chunks_by_path):
    """沿用 golden.jsonl 的檢索結果，依指定版本的 prompt 重組 messages。
    參考資料一律從 data/index/chunks.jsonl 取回完整 chunk 重新排版，與 golden.jsonl 是用哪個版本匯出的無關。"""
    p = PROMPTS[prompt]
    hits = [(chunks_by_path[(h["source"].split("#", 1)[0], tuple(h["path"]))], h["score"]) for h in item["hits"]]
    user = build_user_message(item["question"], hits, p.get("cite_tags", False))
    return [{"role": "system", "content": p["system"]}, {"role": "user", "content": user + p["suffix"]}]


def generate(args, out_path):
    from genai.gemma_taide import GemmaTaide
    from genai.llama_taide import LlamaTaide

    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f}
    items = [i for i in load_items() if (not args.ids or i["id"] in args.ids) and i["id"] not in done]
    if not items:
        print(f"沒有需要生成的題目（{out_path.name} 已完成）")
        return

    chunks_by_path = {(c["doc_id"], tuple(c["path"])): c for c in load_chunks()}
    llm = (LlamaTaide if args.model == "llama" else GemmaTaide)()
    ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as out:
        for n, item in enumerate(items, 1):
            t0 = time.perf_counter()
            messages = build_messages(item, args.prompt, chunks_by_path)
            answer = llm.generate(messages, max_new_tokens=args.max_new_tokens,
                                  do_sample=False, repetition_penalty=1.1)
            seconds = time.perf_counter() - t0
            tokens = len(llm.tokenizer(answer)["input_ids"])
            out.write(json.dumps({"id": item["id"], "question": item["question"], "answer": answer,
                                  "seconds": round(seconds, 1), "tokens": tokens}, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{n}/{len(items)}] {item['id']} {seconds:.0f} 秒｜{answer[:40].replace(chr(10), ' ')}…")


def facts_of(text):
    return {f.rstrip(".,").replace(",", "") for f in FACT.findall(text)}


def score_one(item, answer):
    citations = parse_citations(answer)
    hits = [(h["source"].split("#", 1)[0], h["source"].split("#", 1)[1], h["path"]) for h in item["hits"]]

    def find_hit(doc, sec):
        for h_doc, h_sec, h_path in hits:
            if h_doc == doc and (same_section(h_sec, sec) or any(same_section(p, sec) for p in h_path)):
                return h_path
        return None

    paths, valid = [], 0
    for doc, sec in citations:
        path = find_hit(doc, sec)
        valid += path is not None
        paths.append((doc, path or [sec]))
    cited_expected = any(
        doc == e_doc and any(same_section(h, e_title) for h in path)
        for doc, path in paths
        for e_doc, _, e_title in (s.partition("#") for s in item["expected_sources"])
    )
    expected_facts = facts_of(item["expected_answer"]) if item["answerable"] else set()
    normalized = answer.replace(",", "").replace(" ", "")
    length = len(re.sub(r"\s", "", answer))
    return {
        "id": item["id"], "question_type": item["question_type"], "answerable": item["answerable"],
        "n_citations": len(citations), "valid_citations": valid,
        "has_citation": bool(citations), "cited_expected": cited_expected,
        "fact_recall": (sum(f.replace(" ", "") in normalized for f in expected_facts) / len(expected_facts)
                        if expected_facts else None),
        "standard_refusal": REFUSAL in answer,
        # 用了標準拒答句且整段很短：只有拒答，沒有附帶推測或額外建議
        "short_refusal": REFUSAL in answer and length <= 80,
        "declined": any(m in answer for m in DECLINE_MARKERS),
        "fake_citation": any(doc == "EX-001" for doc, _ in citations),
        "deprecated": any(m in answer for m in DEPRECATED_MARKERS),
        "length": length,
    }


def summarize(scores):
    ans = [s for s in scores if s["answerable"]]
    una = [s for s in scores if not s["answerable"]]
    n_cit = sum(s["n_citations"] for s in ans)
    facts = [s["fact_recall"] for s in ans if s["fact_recall"] is not None]
    mean = lambda xs: sum(xs) / len(xs) if xs else 0
    return {
        "可回答題：有標出處": mean([s["has_citation"] for s in ans]),
        "可回答題：出處有效率": sum(s["valid_citations"] for s in ans) / n_cit if n_cit else 0,
        "可回答題：引用到正確段落": mean([s["cited_expected"] for s in ans]),
        "可回答題：關鍵事實命中": mean(facts),
        "可回答題：誤拒（只回答查不到）": mean([s["short_refusal"] for s in ans]),
        "可回答題：表示資料不足": mean([s["declined"] for s in ans]),
        "無答案題：表示查不到": mean([s["declined"] for s in una]),
        "無答案題：使用標準拒答句": mean([s["standard_refusal"] for s in una]),
        "無答案題：簡潔拒答（≤80 字）": mean([s["short_refusal"] for s in una]),
        "引用虛構範例 EX-001（題數）": sum(s["fake_citation"] for s in scores),
        "引用已廢止內容（題數）": sum(s["deprecated"] for s in scores),
        "平均回答長度（字）": mean([s["length"] for s in scores]),
    }


def score_runs(runs):
    items = {i["id"]: i for i in load_items()}
    results = {}
    for run in runs:
        path = ANSWER_DIR / f"{run}.jsonl"
        with path.open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        scores = [score_one(items[r["id"]], r["answer"]) for r in records]
        results[run] = (records, scores)
        with (ANSWER_DIR / f"{run}_scores.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(scores[0]) + ["answer"])
            w.writeheader()
            for s, r in zip(scores, records):
                w.writerow({**s, "answer": r["answer"]})

    print("\n## 生成評估")
    print(pad("指標", 34) + "".join(f"{run:>14}" for run in runs))
    summaries = {run: summarize(scores) for run, (_, scores) in results.items()}
    for metric in summaries[runs[0]]:
        cells = []
        for run in runs:
            v = summaries[run][metric]
            cells.append(f"{v:>14d}" if isinstance(v, int) else f"{v:>14.1f}" if v > 1 else f"{v:>14.3f}")
        print(pad(metric, 34) + "".join(cells))
    for run, (records, _) in results.items():
        secs = [r["seconds"] for r in records]
        toks = sum(r["tokens"] for r in records)
        print(f"[{run}] {len(records)} 題，平均 {sum(secs) / len(secs):.1f} 秒／題，"
              f"{toks / sum(secs):.1f} tokens/秒 → 每題評分 data/answers/{run}_scores.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", choices=["llama", "gemma"], default="llama")
    parser.add_argument("-p", "--prompt", choices=PROMPTS, default=DEFAULT_PROMPT)
    parser.add_argument("--ids", nargs="+", help="只跑這些題目")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--score", nargs="+", metavar="RUN", help="只評分，如 llama_v1 llama_v2")
    args = parser.parse_args()

    if args.score:
        score_runs(args.score)
        return
    run = f"{args.model}_{args.prompt}"
    generate(args, ANSWER_DIR / f"{run}.jsonl")
    score_runs([run])


if __name__ == "__main__":
    main()

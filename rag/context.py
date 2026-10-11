"""把檢索結果整理成可直接餵給生成模型的內容（messages 或單一 prompt 字串）。

輸出的 messages 格式為 [{"role": "system"/"user", "content": ...}]，
可直接傳給 genai/ 的 LlamaTaide.generate()、GemmaTaide.generate()，或 OpenAI 相容的 API。

用法（在 TAIDE 根目錄執行，需先跑 python -m rag.ingest）:
    python -m rag.context "忘記密碼怎麼辦"           # 印出可直接貼進任何聊天介面的 prompt
    python -m rag.context "忘記密碼怎麼辦" --json    # 印出 JSON（含 messages 與檢索到的段落）
    python -m rag.context --golden                  # 黃金測試集 68 題全部匯出 → data/contexts/golden.jsonl
    python -m rag.context --golden --variants       # 連同口語／錯字變體一起匯出（204 筆）
程式中:
    from rag.context import ContextBuilder
    builder = ContextBuilder()                      # 預設 bge-m3+xcloud、取 5 段
    result = builder.build("忘記密碼怎麼辦")
    reply = llm.generate(result["messages"])
"""
import argparse
import json

from rag.common import GOLDEN_PATH, ROOT
from rag.retrieve import DEFAULT_CONFIG, Retriever

CONTEXT_DIR = ROOT / "data" / "contexts"

SYSTEM_PROMPT = (
    "你是曦光人壽的內部知識庫助理，協助員工查詢公司規章、SOP 與系統操作，一律使用繁體中文回答。\n"
    "回答規則：\n"
    "1. 只能根據使用者訊息中「參考資料」的內容回答，不可使用資料以外的知識或自行推測。\n"
    "2. 每個重點後面標註出處，格式為 [文件編號#章節]，例如 [IT-FAQ-001#Q2. 忘記密碼怎麼辦？]。\n"
    "3. 有步驟的內容請用編號條列。\n"
    "4. 若參考資料中沒有答案，請回答「知識庫中查不到相關資訊」，不要編造。"
)


def format_hits(hits):
    blocks = []
    for i, (chunk, _) in enumerate(hits, 1):
        header = (f"[{i}] 出處：{chunk['doc_id']}#{chunk['section']}\n"
                  f"文件：{chunk['title']}（{chunk['version']}，{chunk['effective_date']} 生效）\n"
                  f"章節：{' > '.join(chunk['path'])}")
        blocks.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(blocks)


def build_user_message(question, hits):
    return f"參考資料：\n\n{format_hits(hits)}\n\n---\n問題：{question}"


class ContextBuilder:
    def __init__(self, config=DEFAULT_CONFIG, k=5, system_prompt=SYSTEM_PROMPT):
        self.retriever = Retriever(config)
        self.k = k
        self.system_prompt = system_prompt

    def build(self, question, history=None):
        """history: 先前的對話 [{"role": "user"/"assistant", ...}]，會放在 system 與本輪問題之間。"""
        hits = self.retriever.search(question, self.k)
        user = build_user_message(question, hits)
        messages = [{"role": "system", "content": self.system_prompt}, *(history or []),
                    {"role": "user", "content": user}]
        return {
            "question": question,
            "messages": messages,
            # 給不支援 system 角色、或要直接貼進聊天介面的情況
            "prompt": f"{self.system_prompt}\n\n{user}",
            "hits": [
                {"rank": i, "score": round(score, 4), "source": f"{c['doc_id']}#{c['section']}",
                 "title": c["title"], "path": c["path"], "text": c["text"]}
                for i, (c, score) in enumerate(hits, 1)
            ],
        }


def export_golden(builder, with_variants):
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    path = CONTEXT_DIR / ("golden_variants.jsonl" if with_variants else "golden.jsonl")
    with GOLDEN_PATH.open(encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    n = 0
    with path.open("w", encoding="utf-8") as out:
        for item in items:
            forms = [("原句", item["question"])]
            if with_variants:
                forms += [("變體", v) for v in item["variants"]]
            for form, question in forms:
                result = builder.build(question)
                record = {"id": item["id"], "form": form, "question_type": item["question_type"],
                          "answerable": item["answerable"], "expected_answer": item["expected_answer"],
                          "expected_sources": item["sources"], **result}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} 筆 → {path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG, help="檢索設定，寫法見 rag/retrieve.py")
    parser.add_argument("-k", type=int, default=5, help="取幾段參考資料")
    parser.add_argument("--json", action="store_true", help="輸出 JSON 而非純文字 prompt")
    parser.add_argument("--golden", action="store_true", help="匯出黃金測試集全部題目")
    parser.add_argument("--variants", action="store_true", help="搭配 --golden，連同變體一起匯出")
    args = parser.parse_args()
    if not args.question and not args.golden:
        parser.error("請給一個問題，或使用 --golden")

    builder = ContextBuilder(args.config, args.k)
    if args.golden:
        export_golden(builder, args.variants)
        return
    result = builder.build(args.question)
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["prompt"])


if __name__ == "__main__":
    main()

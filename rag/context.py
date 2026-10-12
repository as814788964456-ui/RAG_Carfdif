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

REFUSAL = "知識庫中查不到相關資訊"

# 各版本的 system prompt 與附加在問題後的提醒（8B 模型容易忘記放在最前面的規則，問題後再提醒一次）
PROMPTS = {
    "v1": {
        "system": (
            "你是曦光人壽的內部知識庫助理，協助員工查詢公司規章、SOP 與系統操作，一律使用繁體中文回答。\n"
            "回答規則：\n"
            "1. 只能根據使用者訊息中「參考資料」的內容回答，不可使用資料以外的知識或自行推測。\n"
            "2. 每個重點後面標註出處，格式為 [文件編號#章節]，例如 [IT-FAQ-001#Q2. 忘記密碼怎麼辦？]。\n"
            "3. 有步驟的內容請用編號條列。\n"
            f"4. 若參考資料中沒有答案，請回答「{REFUSAL}」，不要編造。"
        ),
        "suffix": "",
    },
    "v2": {
        # 範例使用虛構的 EX-001 文件，避免把測試集內容寫進 prompt
        "system": (
            "你是曦光人壽的內部知識庫助理，只能依據使用者提供的「參考資料」回答員工問題，一律使用繁體中文。\n\n"
            "請依照以下步驟作答：\n"
            "步驟一：判斷參考資料中是否有能直接回答這個問題的內容。\n"
            "步驟二：\n"
            f"- 如果沒有，只輸出這一句，不要加任何說明、推測或建議：{REFUSAL}。\n"
            "- 如果有，依照以下規則作答：\n"
            "  1. 只使用參考資料中寫明的內容與數字，不可加入一般常識、推測，或「通常」「一般來說」「依往例」之類的說法。\n"
            "  2. 每一個重點句子的結尾都要加上出處，格式為 [文件編號#章節]，照抄參考資料中「出處：」後面的文字。\n"
            "  3. 有步驟時用 1. 2. 3. 條列。\n"
            "  4. 不要在回答最後另外列出參考資料清單。\n\n"
            "範例一（參考資料中有答案）：\n"
            "參考資料：[1] 出處：EX-001#三、設備報修　內文：茶水間設備故障請撥總務課分機 1234，報修後 2 個工作天內處理。\n"
            "問題：咖啡機壞了要找誰？\n"
            "回答：請撥總務課分機 1234 報修 [EX-001#三、設備報修]。報修後會在 2 個工作天內處理 [EX-001#三、設備報修]。\n\n"
            "範例二（參考資料中沒有答案）：\n"
            "參考資料：[1] 出處：EX-001#三、設備報修　內文：茶水間設備故障請撥總務課分機 1234，報修後 2 個工作天內處理。\n"
            "問題：公司今年會發三節獎金嗎？\n"
            f"回答：{REFUSAL}。"
        ),
        "suffix": (
            f"\n\n（提醒：參考資料中沒有答案時，只回答「{REFUSAL}。」；"
            "有答案時，每個重點句尾都要標註 [文件編號#章節]。）"
        ),
    },
    "v3": {
        # v2 的問題：模型照抄範例中的虛構文件 EX-001、自行改寫章節名稱、拒答後又加推測。
        # 改為不放範例，參考資料每段直接給出可照抄的引用標註（cite_tags）。
        "system": (
            "你是曦光人壽的內部知識庫助理，只能依據使用者提供的「參考資料」回答員工問題，一律使用繁體中文。\n\n"
            "作答前先判斷：參考資料中是否有能直接回答這個問題的內容？\n\n"
            "【沒有】只輸出下面這一句，前後不要加任何文字、推測或建議：\n"
            f"{REFUSAL}。\n\n"
            "【有】依照以下規則簡潔作答：\n"
            "1. 只使用參考資料中寫明的內容與數字；不可加入一般常識或推測，不可使用「通常」「一般來說」「依往例」等說法。\n"
            "2. 每個重點句子的結尾，加上該內容所在資料的「引用標註」，必須原封不動地複製，包含方括號。\n"
            "3. 有步驟時用 1. 2. 3. 條列。\n"
            "4. 不要在最後另外列出參考資料清單，也不要重複問題。"
        ),
        "suffix": (
            f"\n\n（提醒：先判斷參考資料能否回答這個問題。不能，就只輸出「{REFUSAL}。」；"
            "能，就簡潔作答，並在每個重點句尾複製對應資料的引用標註。）"
        ),
        "cite_tags": True,
    },
}
DEFAULT_PROMPT = "v3"  # 68 題評估中表現最好（見 README 第六節）
SYSTEM_PROMPT = PROMPTS[DEFAULT_PROMPT]["system"]


def format_hits(hits, cite_tags=False):
    """cite_tags=True 時每段開頭直接給出可照抄的「引用標註：[文件編號#章節]」，模型不必自己組出處。"""
    blocks = []
    for i, (chunk, _) in enumerate(hits, 1):
        source = f"{chunk['doc_id']}#{chunk['section']}"
        first = f"資料 {i}｜引用標註：[{source}]" if cite_tags else f"[{i}] 出處：{source}"
        header = (f"{first}\n"
                  f"文件：{chunk['title']}（{chunk['version']}，{chunk['effective_date']} 生效）\n"
                  f"章節：{' > '.join(chunk['path'])}")
        blocks.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(blocks)


def build_user_message(question, hits, cite_tags=False):
    return f"參考資料：\n\n{format_hits(hits, cite_tags)}\n\n---\n問題：{question}"


class ContextBuilder:
    def __init__(self, config=DEFAULT_CONFIG, k=5, prompt=DEFAULT_PROMPT, embedders=None):
        """prompt：PROMPTS 中的版本名稱（v1、v2、v3）。
        embedders：{模型名稱: embedder}，可傳入在指定裝置（如 CPU）上已載入的模型。"""
        self.retriever = Retriever(config, embedders=embedders)
        self.k = k
        self.system_prompt = PROMPTS[prompt]["system"]
        self.suffix = PROMPTS[prompt]["suffix"]
        self.cite_tags = PROMPTS[prompt].get("cite_tags", False)

    def build(self, question, history=None, search_query=None):
        """history: 先前的對話 [{"role": "user"/"assistant", ...}]，會放在 system 與本輪問題之間。
        search_query: 用來檢索的查詢（字串，或多個查詢的 list，各自檢索後融合），預設就是 question。"""
        hits = self.retriever.search(search_query or question, self.k)
        user = build_user_message(question, hits, self.cite_tags) + self.suffix
        messages = [{"role": "system", "content": self.system_prompt}, *(history or []),
                    {"role": "user", "content": user}]
        return {
            "question": question,
            "messages": messages,
            # 給不支援 system 角色、或要直接貼進聊天介面的情況
            "prompt": f"{self.system_prompt}\n\n{user}",
            "hits": [
                {"rank": i, "score": round(score, 4), "chunk_id": c["id"], "source": f"{c['doc_id']}#{c['section']}",
                 "doc_id": c["doc_id"], "section": c["section"], "title": c["title"], "version": c["version"],
                 "effective_date": c["effective_date"], "file": c["file"], "path": c["path"], "text": c["text"]}
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
    parser.add_argument("-p", "--prompt", choices=PROMPTS, default=DEFAULT_PROMPT, help="prompt 版本")
    parser.add_argument("--json", action="store_true", help="輸出 JSON 而非純文字 prompt")
    parser.add_argument("--golden", action="store_true", help="匯出黃金測試集全部題目")
    parser.add_argument("--variants", action="store_true", help="搭配 --golden，連同變體一起匯出")
    args = parser.parse_args()
    if not args.question and not args.golden:
        parser.error("請給一個問題，或使用 --golden")

    builder = ContextBuilder(args.config, args.k, args.prompt)
    if args.golden:
        export_golden(builder, args.variants)
        return
    result = builder.build(args.question)
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["prompt"])


if __name__ == "__main__":
    main()

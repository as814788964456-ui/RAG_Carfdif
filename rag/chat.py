"""RAG 問答對話：檢索知識庫 → TAIDE 依參考資料回答 → 列出引用的出處。

用法（在 TAIDE 根目錄執行，需先跑 python -m rag.ingest）:
    python -m rag.chat                 # Llama-3.1-TAIDE 8B，prompt v3，bge-m3+xcloud 檢索
    python -m rag.chat -k 3            # 只給模型前 3 段參考資料（回答較快）
對話中的指令:
    /src    顯示上一題檢索到的參考資料
    /reset  清除對話歷史
    /exit   離開

顯卡只放 TAIDE（生成時約 6.8 GiB），檢索用的 embedding 模型一律在 CPU 執行。
多輪對話：追問（如「那要怎麼申請？」「熱點也不行」）時，同時用「本輪問題」與「前一輪問題＋本輪問題」
檢索並融合結果，讓指代前文的問題也找得到資料；模型另外會看到最近幾輪的問答。
"""
import argparse
import re
import sys
import time

from rag.answer import parse_citations, same_section
from rag.context import DEFAULT_PROMPT, PROMPTS, REFUSAL, ContextBuilder
from rag.common import get_embedder
from rag.retrieve import DEFAULT_CONFIG, parse_config

# 指代前文的說法；問題很短時也多半是在追問
FOLLOW_UP = re.compile(r"^(那|他|她|它|這|還是|也|如果|所以|然後|另外|改|換)|它|他們|上述|剛剛|剛才|前面|同樣")


def is_follow_up(question):
    return len(question) <= 10 or bool(FOLLOW_UP.search(question))


def search_query(question, prev_question):
    if prev_question and is_follow_up(question):
        return [question, f"{prev_question} {question}"]
    return question


def cited_hits(answer, hits):
    """依回答中出現的順序，找出被引用的參考資料（去除重複）。"""
    out = []
    for doc, sec in parse_citations(answer):
        for h in hits:
            if h["doc_id"] == doc and (same_section(h["section"], sec) or any(same_section(p, sec) for p in h["path"])):
                if h not in out:
                    out.append(h)
                break
    return out


def format_source(h):
    return (f"{h['doc_id']}《{h['title']}》{h['version']}（{h['effective_date']} 生效）"
            f"＞{' > '.join(h['path'])}｜{h['file']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", choices=["llama", "gemma"], default="llama")
    parser.add_argument("-p", "--prompt", choices=PROMPTS, default=DEFAULT_PROMPT)
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG, help="檢索設定，寫法見 rag/retrieve.py")
    parser.add_argument("-k", type=int, default=5, help="給模型幾段參考資料")
    parser.add_argument("--history", type=int, default=2, help="模型看得到的最近對話輪數")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    import warnings

    from transformers import TextStreamer
    from transformers.utils import logging as hf_logging

    # 對話畫面只顯示問答；transformers 的生成參數提醒等訊息會插在回答中間
    hf_logging.set_verbosity_error()
    warnings.filterwarnings("ignore")

    if args.model == "llama":
        from genai.llama_taide import LlamaTaide as Model
    else:
        from genai.gemma_taide import GemmaTaide as Model

    print("載入檢索模型（CPU）…")
    embedders = {name: get_embedder(name, device="cpu") for name, _ in parse_config(args.config) if name != "bm25"}
    builder = ContextBuilder(args.config, args.k, args.prompt, embedders)
    print(f"載入 {Model.repo_id} …")
    llm = Model()
    streamer = TextStreamer(llm.tokenizer, skip_prompt=True, skip_special_tokens=True)
    piped = not sys.stdin.isatty()  # 從檔案或管線輸入問題時，把問題也印出來

    history, prev_question, last_hits = [], None, []
    print("\n完成。直接輸入問題；/src 看參考資料、/reset 清除對話、/exit 離開。\n")
    while True:
        try:
            question = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if piped:
            print(question)
        if not question:
            continue
        if question == "/exit":
            break
        if question == "/reset":
            history, prev_question, last_hits = [], None, []
            print("（已清除對話歷史）\n")
            continue
        if question == "/src":
            for h in last_hits:
                print(f"  [{h['rank']}] {format_source(h)}\n      {h['text'][:120].replace(chr(10), ' ')}…")
            print()
            continue

        t0 = time.perf_counter()
        query = search_query(question, prev_question)
        result = builder.build(question, history=history[-2 * args.history:], search_query=query)
        t1 = time.perf_counter()
        print("TAIDE> ", end="", flush=True)
        answer = llm.generate(result["messages"], max_new_tokens=args.max_new_tokens, streamer=streamer,
                              do_sample=False, repetition_penalty=1.1).strip()
        t2 = time.perf_counter()

        cited = cited_hits(answer, result["hits"])
        if cited:
            print("\n出處：")
            for h in cited:
                print(f"  • {format_source(h)}")
        elif REFUSAL in answer:
            print("\n（知識庫中沒有相關資料。可以換個問法，或洽詢承辦單位，分機與窗口見 GEN-002《分機與聯絡窗口》）")
        else:
            print("\n（模型沒有標註出處，以下是檢索到的前 3 段資料）")
            for h in result["hits"][:3]:
                print(f"  • {format_source(h)}")
        follow = "，追問：合併前一題檢索" if isinstance(query, list) else ""
        print(f"（檢索 {t1 - t0:.1f} 秒{follow}；生成 {t2 - t1:.1f} 秒）\n")

        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        prev_question, last_hits = question, result["hits"]


if __name__ == "__main__":
    main()

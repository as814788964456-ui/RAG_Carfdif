"""TAIDE 模型命令列對話。

用法:
    python chat.py llama           # Llama-3.1-TAIDE-LX-8B-Chat
    python chat.py gemma           # Gemma-3-TAIDE-12b-Chat-2602
    python chat.py gemma --no-4bit # 不量化（需要足夠 VRAM）

對話中輸入 /reset 清除歷史，/exit 離開。
"""
import argparse

from transformers import TextStreamer

from gemma_taide import GemmaTaide
from llama_taide import SYSTEM_PROMPT, LlamaTaide

MODELS = {
    "llama": LlamaTaide,
    "gemma": GemmaTaide,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--no-4bit", action="store_true", help="不使用 4-bit 量化")
    parser.add_argument("--gpu-mem", help="每張 GPU 最多使用的記憶體，如 6GiB（預設：顯卡記憶體減 2GiB）")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    cls = MODELS[args.model]
    print(f"載入 {cls.repo_id} ...")
    llm = cls(not args.no_4bit, args.gpu_mem)
    streamer = TextStreamer(llm.tokenizer, skip_prompt=True, skip_special_tokens=True)

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("完成。輸入 /reset 清除歷史，/exit 離開。\n")
    while True:
        try:
            user = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            history = history[:1]
            print("（已清除對話歷史）\n")
            continue

        history.append({"role": "user", "content": user})
        print("TAIDE> ", end="", flush=True)
        reply = llm.generate(history, max_new_tokens=args.max_new_tokens, streamer=streamer)
        history.append({"role": "assistant", "content": reply})
        print()


if __name__ == "__main__":
    main()

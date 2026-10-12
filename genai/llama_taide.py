"""Llama-3.1-TAIDE-LX-8B-Chat（約 16 GB，4-bit 量化後約 5.5 GB VRAM）。

用法:
    python llama_taide.py "問題"            # 單次問答
    python llama_taide.py "問題" --no-4bit  # 不量化（需要足夠 VRAM）
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hardware import load_kwargs

SYSTEM_PROMPT = "你是一個來自台灣的AI助理，你的名字是 TAIDE，樂於以台灣人的立場幫助使用者，會用繁體中文回答問題。"


class LlamaTaide:
    name = "llama"
    vram_gib_4bit = 6.8  # 4-bit 時約需的顯卡記憶體（實測權重 6.2 GiB＋推論空間）
    repo_id = "taide/Llama-3.1-TAIDE-LX-8B-Chat"

    def __init__(self, use_4bit=True, gpu_mem=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
        kwargs = load_kwargs(use_4bit, gpu_mem, self.vram_gib_4bit)
        self.model = AutoModelForCausalLM.from_pretrained(self.repo_id, **kwargs)
        self.model.eval()

    def generate(self, messages, max_new_tokens=1024, streamer=None, **gen_kwargs):
        """messages 為 [{"role": ..., "content": ...}, ...]，回傳模型回覆文字。

        gen_kwargs 可覆蓋取樣參數，例如 do_sample=False（每次輸出相同，評估時使用）。
        """
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        params = {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.1, **gen_kwargs}
        if not params["do_sample"]:
            params.pop("temperature", None)
            params.pop("top_p", None)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                **params,
            )
        reply = self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return reply.strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--no-4bit", action="store_true", help="不使用 4-bit 量化")
    parser.add_argument("--gpu-mem", help="不量化時每張 GPU 最多使用的記憶體，如 6GiB（預設：剩餘記憶體減 0.5GiB）")
    args = parser.parse_args()

    llm = LlamaTaide(not args.no_4bit, args.gpu_mem)
    print(llm.generate([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": args.question},
    ]))

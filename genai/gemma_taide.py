"""Gemma-3-TAIDE-12b-Chat-2602（約 24 GB，4-bit 量化後約 7–8 GB，VRAM 不夠的部分會放到 CPU）。

用法:
    python gemma_taide.py "問題"            # 單次問答
    python gemma_taide.py "問題" --no-4bit  # 不量化（需要足夠 VRAM）
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hardware import load_kwargs

SYSTEM_PROMPT = "你是一個來自台灣的AI助理，你的名字是 TAIDE，樂於以台灣人的立場幫助使用者，會用繁體中文回答問題。"


class GemmaTaide:
    name = "gemma"
    repo_id = "taide/Gemma-3-TAIDE-12b-Chat-2602"

    def __init__(self, use_4bit=True, gpu_mem=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
        kwargs = load_kwargs(use_4bit, gpu_mem)
        if kwargs["dtype"] == torch.float16:
            print("警告：此顯卡不支援 bf16，Gemma 在 fp16 下可能產生 NaN（回答空白或亂碼）")
        self.model = AutoModelForCausalLM.from_pretrained(self.repo_id, **kwargs)
        self.model.eval()

    def generate(self, messages, max_new_tokens=1024, streamer=None):
        """messages 為 [{"role": ..., "content": ...}, ...]，回傳模型回覆文字。"""
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                streamer=streamer,
            )
        reply = self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return reply.strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--no-4bit", action="store_true", help="不使用 4-bit 量化")
    parser.add_argument("--gpu-mem", help="每張 GPU 最多使用的記憶體，如 6GiB（預設：顯卡記憶體減 2GiB）")
    args = parser.parse_args()

    llm = GemmaTaide(not args.no_4bit, args.gpu_mem)
    print(llm.generate([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": args.question},
    ]))

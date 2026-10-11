"""依執行環境（本機 / Colab / Kaggle）自動決定模型載入參數。

- dtype：顯卡支援 bf16（Ampere 以後，如 RTX 30 系列、L4、A100）用 bf16，否則（如 T4、P100）用 fp16。
- max_memory：每張 GPU 預留 2GiB 給推論時的 KV cache；CPU 用目前可用記憶體的 80%。
"""
import psutil
import torch
from transformers import BitsAndBytesConfig

GiB = 1024**3
GPU_RESERVE_GIB = 2


def pick_dtype():
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    return torch.float16


def load_kwargs(use_4bit=True, gpu_mem=None):
    """回傳給 AutoModelForCausalLM.from_pretrained 的參數。gpu_mem 為 None 時自動計算。"""
    if not torch.cuda.is_available():
        print("找不到 GPU，改用 CPU 執行（非常慢，且無法 4-bit 量化）")
        return {"device_map": "cpu", "dtype": torch.float32}

    dtype = pick_dtype()
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory / GiB
        max_memory[i] = gpu_mem or f"{int(max(total - GPU_RESERVE_GIB, 1) * 1024)}MiB"
    max_memory["cpu"] = f"{max(int(psutil.virtual_memory().available / GiB * 0.8), 1)}GiB"

    kwargs = {"device_map": "auto", "dtype": dtype, "max_memory": max_memory}
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            # VRAM 不夠時允許部分層放到 CPU（會變慢）
            llm_int8_enable_fp32_cpu_offload=True,
        )
    print(f"{torch.cuda.get_device_name(0)}，dtype={dtype}，max_memory={max_memory}")
    return kwargs

"""依執行環境（本機 / Colab / Kaggle）自動決定模型載入參數。

- dtype：顯卡支援 bf16（Ampere 以後，如 RTX 30 系列、L4、A100）用 bf16，否則（如 T4、P100）用 fp16。
- 4-bit 量化：整個模型必須放在 GPU 上。bitsandbytes 0.50 的 4-bit 層一旦被分到 CPU，
  載入或生成時會出現 meta tensor 錯誤，所以載入前先檢查「目前剩餘」的顯卡記憶體，
  不夠就直接提示，而不是分一部分到 CPU。
- 不量化（bf16/fp16）：放不下的層可以分到 CPU（會變慢）。
"""
import psutil
import torch
from transformers import BitsAndBytesConfig

GiB = 1024**3
GPU_RESERVE_GIB = 0.5  # 不量化時，每張 GPU 在剩餘記憶體之外再保留給推論的空間


def pick_dtype():
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    return torch.float16


def free_gib(device=0):
    """目前實際剩餘的顯卡記憶體（已扣除遊戲、瀏覽器等其他程式佔用的部分）。"""
    return torch.cuda.mem_get_info(device)[0] / GiB


def load_kwargs(use_4bit=True, gpu_mem=None, required_gib=None):
    """回傳給 AutoModelForCausalLM.from_pretrained 的參數。

    required_gib：4-bit 時模型（含推論空間）大約需要的顯卡記憶體，用來在載入前檢查。
    gpu_mem：不量化時每張 GPU 的上限，如 "6GiB"；None 則用目前剩餘記憶體自動計算。
    """
    if not torch.cuda.is_available():
        print("找不到 GPU，改用 CPU 執行（非常慢，且無法 4-bit 量化）")
        return {"device_map": "cpu", "dtype": torch.float32}

    dtype = pick_dtype()
    n_gpu = torch.cuda.device_count()
    free = [free_gib(i) for i in range(n_gpu)]
    total = torch.cuda.get_device_properties(0).total_memory / GiB
    print(f"{torch.cuda.get_device_name(0)}，dtype={dtype}，剩餘顯卡記憶體 "
          + "、".join(f"{f:.1f}/{total:.1f} GiB" for f in free))

    if use_4bit:
        available = free[0] if n_gpu == 1 else sum(free)
        if required_gib and available < required_gib:
            raise RuntimeError(
                f"顯卡剩餘 {available:.1f} GiB，4-bit 模型約需要 {required_gib:.1f} GiB。\n"
                "請先關閉佔用顯卡的程式（遊戲、瀏覽器分頁、Discord、Steam 等）後再試，"
                "可用 nvidia-smi 查看剩餘記憶體；或改到 Colab／Kaggle 執行。"
            )
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        # 單張 GPU 直接整個放在 GPU 0；多張 GPU（如 Kaggle 2×T4）讓 accelerate 分配，但不允許放到 CPU
        device_map = {"": 0} if n_gpu == 1 else "auto"
        kwargs = {"device_map": device_map, "dtype": dtype, "quantization_config": quant}
        if n_gpu > 1:
            kwargs["max_memory"] = {i: f"{int(f * 1024)}MiB" for i, f in enumerate(free)}
        return kwargs

    max_memory = {i: gpu_mem or f"{int(max(f - GPU_RESERVE_GIB, 0.5) * 1024)}MiB" for i, f in enumerate(free)}
    max_memory["cpu"] = f"{max(int(psutil.virtual_memory().available / GiB * 0.8), 1)}GiB"
    print(f"max_memory={max_memory}")
    return {"device_map": "auto", "dtype": dtype, "max_memory": max_memory}

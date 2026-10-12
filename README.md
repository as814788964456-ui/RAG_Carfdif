# RAG_Carfdif：TAIDE 本機 RAG 知識庫問答

以台灣本土模型 **TAIDE** 為生成模型、**bge-m3／xVITA-Embed-zhTW** 為 embedding 模型，在本機（RTX 3070 8GB）建立保險公司內部知識庫（規章、SOP、系統手冊）的 RAG 問答系統。知識庫使用「曦光人壽」模擬資料（全部為虛構，見 [data/knowledge-base/README.md](data/knowledge-base/README.md)）。

## 目錄

1. [目前進度](#一目前進度)
2. [專案結構](#二專案結構)
3. [環境建置](#三環境建置)
4. [快速開始](#四快速開始)
5. [已完成部分與詳細過程](#五已完成部分與詳細過程)
6. [評估結果](#六評估結果)
7. [待辦事項](#七待辦事項)
8. [在 Colab／Kaggle 上執行](#八在-colabkaggle-上執行)

---

## 一、目前進度

RAG = 檢索（Retrieval）＋ 生成（Generation）。**檢索端已完成並經過量化評估；生成端已準備好輸入，尚未實際生成與評估。**

| 階段 | 狀態 | 說明 |
|---|---|---|
| 生成模型模組 | ✅ | Llama-3.1-TAIDE 8B、Gemma-3-TAIDE 12B，4-bit 量化，自動偵測硬體 |
| Embedding 模組 | ✅ | bge-m3（GPU）、xVITA-Embed-zhTW（CPU） |
| 切塊與建索引 | ✅ | 39 份文件依章節切成 300 塊，保留 metadata |
| 檢索 | ✅ | 兩個 embedding 模型以 RRF 融合，Hit@3 = 0.956 |
| 檢索評估 | ✅ | 黃金測試集 68 題（含口語／錯字變體共 204 次查詢） |
| 已廢止文件過濾 | ✅ | `status: 已廢止` 一律不進檢索結果 |
| Prompt 組裝與匯出 | ✅ | `ContextBuilder`；68 題已匯出成 `data/contexts/golden.jsonl` |
| TAIDE 實際生成答案 | 🟡 | 已在 3070 上跑通，約 15 tokens/秒；只試過 3 題知識庫問題（見 [6.5](#65-生成初步測試3-題)） |
| 生成品質評估 | ❌ | 初步發現：內容正確但不照格式標出處、無答案題會編造；尚未完整量測 |
| 可互動的 RAG 對話程式 | ❌ | |

## 二、專案結構

```
RAG_Carfdif/
├─ genai/                     生成模型
│  ├─ llama_taide.py          LlamaTaide：Llama-3.1-TAIDE-LX-8B-Chat
│  ├─ gemma_taide.py          GemmaTaide：Gemma-3-TAIDE-12b-Chat-2602
│  ├─ hardware.py             依硬體自動決定 dtype、量化、記憶體上限
│  └─ chat.py                 命令列多輪對話（尚未接 RAG）
├─ embedding/                 Embedding 模型
│  ├─ bge_m3.py               BgeM3Embedder（1024 維）
│  ├─ xcloud.py               XCloudEmbedder（768 維，GGUF／llama.cpp）
│  ├─ search_demo.py          兩個模型的檢索比較示範
│  └─ sample_docs.json        示範用文件
├─ rag/                       RAG 流程
│  ├─ common.py               路徑、模型清單、索引讀取
│  ├─ ingest.py               切塊＋建向量索引
│  ├─ retrieve.py             語意／BM25／RRF 融合檢索
│  ├─ eval.py                 檢索評估（Hit@k、MRR、Recall@5）
│  └─ context.py              把檢索結果組成 prompt／messages
├─ data/
│  ├─ knowledge-base/         知識庫 39 份文件＋黃金測試集
│  ├─ contexts/golden.jsonl   68 題的檢索結果與 prompt（可直接餵給生成模型）
│  ├─ index/                  （不進 repo，由 rag.ingest 產生）
│  └─ eval/                   （不進 repo，由 rag.eval 產生）
└─ requirements.txt
```

## 三、環境建置

### 1. 建立 Python 環境（Python 3.12）

```powershell
git clone https://github.com/as814788964456-ui/RAG_Carfdif.git
cd RAG_Carfdif
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` 中的 torch 為 CUDA 12.8 版、llama-cpp-python 為 CPU 版。

### 2. 取得 TAIDE 模型授權

TAIDE 為 gated 模型，需要先：

1. 登入 Hugging Face，到兩個模型頁面按同意授權條款（自動核准）：
   [Llama-3.1-TAIDE-LX-8B-Chat](https://huggingface.co/taide/Llama-3.1-TAIDE-LX-8B-Chat)、[Gemma-3-TAIDE-12b-Chat-2602](https://huggingface.co/taide/Gemma-3-TAIDE-12b-Chat-2602)
2. 到 https://huggingface.co/settings/tokens 建立 Read token
3. `hf auth login` 貼上 token

Embedding 模型不需要授權，第一次使用時自動下載。模型預設存在 `%USERPROFILE%\.cache\huggingface\hub`，要換位置可設定 `$env:HF_HUB_CACHE = "D:\hf_cache"`。

### 3. 使用的模型

| 用途 | 簡稱 | 模型 | 大小 | 執行方式 |
|---|---|---|---|---|
| 生成 | `llama` | [taide/Llama-3.1-TAIDE-LX-8B-Chat](https://huggingface.co/taide/Llama-3.1-TAIDE-LX-8B-Chat) | 約 16 GB（4-bit 實測 6.2 GiB，生成時峰值 6.8 GiB） | transformers＋bitsandbytes |
| 生成 | `gemma` | [taide/Gemma-3-TAIDE-12b-Chat-2602](https://huggingface.co/taide/Gemma-3-TAIDE-12b-Chat-2602) | 約 24 GB（4-bit 約 8 GiB，**8 GB 顯卡放不下**，需用 Colab） | 同上 |
| Embedding | `bge-m3` | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | 1024 維 | sentence-transformers（GPU fp16） |
| Embedding | `xcloud` | [xCloudinfo/xVITA-Embed-300M-zhTW-GGUF](https://huggingface.co/xCloudinfo/xVITA-Embed-300M-zhTW-GGUF)（`xcloud-emb-zhtw-Q8_0.gguf`） | 768 維 | llama-cpp-python（CPU） |
| Embedding | `xcloud-8k` | 同上長文版（`xcloud-emb-zhtw-8k-Q8_0.gguf`） | 768 維 | 同上（尚未評估） |

## 四、快速開始

RAG 指令都在 repo 根目錄執行：

```powershell
python -m rag.ingest                              # 1. 切塊並建立 bge-m3、xcloud 索引 → data/index/
python -m rag.retrieve "忘記密碼怎麼辦"             # 2. 檢索（預設 bge-m3+xcloud 融合）
python -m rag.eval                                # 3. 檢索評估 → data/eval/
python -m rag.context "忘記密碼怎麼辦"              # 4. 印出可直接貼給任何 LLM 的完整 prompt
python -m rag.context --golden                    #    或把 68 題全部匯出 → data/contexts/golden.jsonl
```

把匯出的 prompt 交給 TAIDE 生成（此步驟不需要 embedding 模型，可分開在 Colab 執行）：

```python
import json, sys
sys.path.insert(0, "genai")
from llama_taide import LlamaTaide

llm = LlamaTaide()
for line in open("data/contexts/golden.jsonl", encoding="utf-8"):
    item = json.loads(line)
    print(item["id"], llm.generate(item["messages"]))
```

即時問答：

```python
from rag.context import ContextBuilder
result = ContextBuilder().build("忘記密碼怎麼辦")   # 回傳 messages、prompt、hits
reply = llm.generate(result["messages"])
```

單獨使用生成模型（在 `genai/` 目錄）：

```powershell
cd genai
python chat.py llama                           # 互動式多輪對話（/reset 清除、/exit 離開）
python llama_taide.py "台灣最高的山是哪一座？"     # 單次問答
python gemma_taide.py "問題" --gpu-mem 7GiB      # 手動指定 GPU 記憶體上限
```

---

## 五、已完成部分與詳細過程

### 5.1 生成模型模組（`genai/`）

1. **兩個模型各自一個檔案**：`LlamaTaide`、`GemmaTaide` 介面相同，`generate(messages, max_new_tokens=1024, streamer=None)` 接收 `[{"role", "content"}]` 並回傳回覆文字。取樣參數：temperature 0.7、top_p 0.9、repetition_penalty 1.1。
2. **4-bit 量化**：bitsandbytes NF4＋double quant。8B 不量化需約 16 GB；量化後實測權重 6.2 GiB、生成時峰值 6.8 GiB。比一般估計的 5.5 GB 多，是因為 Llama 3.1 詞彙表很大（128K），輸入嵌入層與輸出層不量化就各佔約 1 GB。
   - **4-bit 模型必須整個放在 GPU 上**：bitsandbytes 0.50＋transformers 5.19 的組合下，4-bit 層只要被分到 CPU，載入時會出現 `Tensor.item() cannot be called on meta tensors`；關閉 double quant 後能載入，但生成時改出現 `Cannot copy out of meta tensor`。
   - 因此 3070 上 Llama 8B 需要約 **7 GB 剩餘顯卡記憶體**，執行前要關閉遊戲等佔用顯卡的程式（實測遊戲執行時剩餘僅 5.1 GB，且兩者搶記憶體會使生成慢到無法使用）。Gemma 12B 在 8 GB 顯卡上無法執行。
   - 實測（3070，無其他程式佔用）：載入 31 秒（第一次約 73 秒），生成約 **15 tokens/秒**，一段 200 tokens 的回答約 13 秒。
3. **自動偵測硬體（`hardware.py`）**，讓同一份程式在本機、Colab、Kaggle 都能直接執行：
   - dtype：顯卡 compute capability ≥ 8（RTX 30 系列、L4、A100）用 bf16，否則（T4、P100）用 fp16。不用 `torch.cuda.is_bf16_supported()`，因為新版 torch 在 T4 上會因「可模擬」而回傳 True。
   - 以**目前實際剩餘**的顯卡記憶體（`torch.cuda.mem_get_info`）判斷，而不是顯卡總量，已扣除遊戲、瀏覽器等其他程式的佔用。
   - 4-bit：單張 GPU 時整個模型放在 GPU 0；載入前比對各模型的需求（`vram_gib_4bit`：Llama 6.8、Gemma 9.0 GiB），不夠就直接提示要關閉哪些程式，而不是分到 CPU 後出錯。多張 GPU（Kaggle 2×T4）時由 accelerate 分配到各張 GPU，但不放到 CPU。
   - 不量化（`--no-4bit`）：放不下的層可分到 CPU，CPU 上限為當下可用記憶體的 80%。
   - Gemma 在 fp16 下容易產生 NaN，偵測到 fp16 時會提出警告。
4. `chat.py` 為命令列多輪對話，支援串流輸出。

### 5.2 Embedding 模組（`embedding/`）

1. **兩個模型各自一個檔案**，介面相同：`embed_queries(list[str])`、`embed_documents(list[str])`，回傳 L2 正規化後的 numpy 陣列，相似度直接用內積（＝cosine）。
2. **bge-m3**：sentence-transformers 載入，GPU 上用 fp16，查詢與文件不需前綴。
3. **xVITA-Embed-zhTW（xcloud）**：GGUF 格式，以 llama-cpp-python 在 CPU 執行；EmbeddingGemma 架構，查詢要加 `task: search result | query: `、文件要加 `title: none | text: `，程式已自動處理。
4. **效能修正**：原本一次把整批文字傳給 `Llama.embed()`，llama.cpp 會把多段文字打包進同一個 8192-token 的 ubatch，注意力運算量隨總長度平方成長。300 塊花了約 10 分鐘、佔用 12.9 GB 記憶體。改為逐筆編碼後速度提升一倍以上，向量與原本幾乎相同（cosine ≥ 0.9997）。調小 `n_ctx` 則幾乎沒有影響。

### 5.3 知識庫資料（`data/knowledge-base/`）

- **39 份 Markdown 文件**，分 10 類：公司總覽、IT FAQ、系統手冊、人事規章、行政財務、保險業務流程 SOP、專業術語、教育訓練、表單、專案經驗，共約 13.6 萬字。
- 每份文件開頭有 YAML front matter：`doc_id`、`title`、`department`、`category`、`doc_type`、`version`、`effective_date`、`status`（生效中／已廢止）、`confidentiality`、`keywords`。
- **黃金測試集** `test-set/golden_set.jsonl`：68 題，每題附 2 個口語／錯字變體；`sources` 以 `doc_id#章節標題` 標示應檢索到的段落。題型：single 51、multi_doc（跨文件）7、version_conflict（版本衝突）3、unanswerable（無答案）7。
- 刻意設計的陷阱：一份已廢止的舊版 VPN 文件（`IT-FAQ-002-OLD`）、跨文件題、無答案題、口語與錯字。

### 5.4 切塊與建索引（`rag/ingest.py`）

1. **讀 front matter**：以 `---` 分隔並用 PyYAML 解析，日期轉為字串。
2. **依章節切塊**：每個 `##`、`###` 標題為一塊，記錄章節路徑（例如 `三、應備文件 > 小額醫療理賠簡化作業`）。
   - 程式碼區塊（```）內的 `#` 不視為標題。
   - 第一個 `##` 之前的前言（H1 與虛構資料聲明）不收。
   - 只有子標題、本身沒有內文的 `##` 不另成一塊（由子塊的章節路徑涵蓋）。
   - 章節長度中位數約 120 字、最長約 1,200 字（≤ 825 tokens），不需要再細切。
3. **結果**：39 份文件 → **300 塊**。每塊保存 front matter 欄位、`section`、`path`、`file`、`text`（原文），以及送進 embedding 的 `content`＝「文件標題｜章節路徑」＋原文（讓短段落也有上下文）。
4. **向量索引**：直接存成 numpy（`data/index/<模型>.npy`），與 `chunks.jsonl` 順序對應。300 塊的規模用內積即可在毫秒內完成檢索，不需要向量資料庫。建索引時間：bge-m3 約 25 秒（GPU），xcloud 約 2 分鐘（121 秒，CPU）。

### 5.5 檢索（`rag/retrieve.py`）

1. **語意檢索**：查詢向量與所有 chunk 向量做內積。
2. **已廢止文件過濾**：`status: 已廢止` 的 chunk 分數設為 -∞。若不過濾，204 次查詢中有 9–11 次會把舊版 VPN 文件排進前 3 名。
3. **BM25 關鍵字檢索**（自行實作，不需額外套件）：
   - 斷詞：英數字取整個詞（`LIFE-Core`、`P1`、`vip`），中文取相鄰兩字（bigram）。不需要斷詞詞典，錯字也只影響兩個 bigram。
   - **查詢去除口語虛詞**（什麼、怎麼、如何、可以、嗎…）：文件是公文語氣，很少出現這些字，BM25 會誤判為稀有關鍵字而壓過真正的關鍵字（例如「VIP件是什麼」原本排第一的是「密碼規則是什麼？」）。去除後 BM25 Hit@1 由 0.514 提升到 0.552。
   - 長度正規化參數 b 在 0.3–0.75 之間差異不大，保留標準值 0.75。
4. **RRF 融合**：多個來源各自排名後以 `權重 / (60 + 名次)` 加總。檢索設定以 `+` 組合，例如 `bge-m3+xcloud`、`xcloud+bm25*0.3`。
5. **實驗與決策**：
   - BM25 加入融合（權重 1.0、0.5、0.3、0.2）整體都變差或持平；BM25 單獨只有 Hit@1 0.55，會把語意檢索排對的結果擠下去。它只在縮寫題（「VIP件」）有幫助。
   - **兩個 embedding 模型互相融合效果最好**（Hit@1 0.809 → 0.847），且沒有需要調整的權重，較不會過度擬合測試集 → **設為預設**。

### 5.6 檢索評估（`rag/eval.py`）

- **評估對象**：61 題可回答題 × 3 種問法（原句＋2 個變體）＝ 183 次查詢；7 題無答案題另外看相似度分佈。
- **命中判定**：chunk 的 `doc_id` 相同，且章節路徑中有標題對應到 `sources` 的章節（`IT-FAQ-001#Q2` 對應「Q2. 忘記密碼怎麼辦？」，但不對應「Q20…」）。執行時會先確認所有 `sources` 都找得到對應的 chunk。
- **指標**：Hit@1／3／5（正確段落是否在前 k 名）、MRR@10、Recall@5（跨文件題需要的多個出處中，前 5 名涵蓋了幾成）。
- 依問法、題型、類別分組輸出，每題前 3 名結果另存 CSV 方便錯誤分析。

### 5.7 Prompt 組裝與匯出（`rag/context.py`）

- `ContextBuilder(config="bge-m3+xcloud", k=5).build(問題, history=None)` 回傳：
  - `messages`：system（回答規則）＋ 先前對話 ＋ user（參考資料＋問題），可直接交給 `LlamaTaide.generate()` 或 OpenAI 相容 API
  - `prompt`：同內容合併成單一字串，可直接貼進任何聊天介面
  - `hits`：檢索到的段落、出處與分數
- **System prompt 的回答規則**：只能根據參考資料回答；每個重點標註出處 `[文件編號#章節]`；步驟用編號條列；查不到時回答「知識庫中查不到相關資訊」。
- 每段參考資料附出處、文件名稱、版本、生效日與章節路徑。
- `--golden` 匯出 68 題（`--variants` 含變體共 204 筆），每筆附 `expected_answer`、`expected_sources` 方便比對。最長的 prompt 約 3,200 字。

---

## 六、評估結果

可回答題 61 題 × 3 種問法 ＝ 183 次查詢，已排除已廢止文件。

### 整體

| 檢索設定 | Hit@1 | Hit@3 | Hit@5 | MRR | Recall@5 |
|---|---|---|---|---|---|
| bge-m3 | 0.809 | 0.929 | 0.962 | 0.874 | 0.857 |
| xcloud | 0.803 | 0.940 | 0.962 | 0.875 | 0.862 |
| bm25 | 0.552 | 0.776 | 0.825 | 0.669 | 0.709 |
| bge-m3 + bm25（權重 1.0） | 0.732 | 0.880 | 0.934 | 0.818 | 0.824 |
| **bge-m3 + xcloud（預設）** | **0.847** | **0.956** | **0.973** | **0.901** | **0.874** |

### 依問法

| 檢索設定 | 原句 Hit@1 | 原句 Hit@3 | 變體 Hit@1 | 變體 Hit@3 |
|---|---|---|---|---|
| bge-m3 | 0.852 | 0.934 | 0.787 | 0.926 |
| xcloud | 0.885 | 0.951 | 0.762 | 0.934 |
| bm25 | 0.623 | 0.885 | 0.516 | 0.721 |
| **bge-m3 + xcloud** | **0.885** | **0.984** | **0.828** | **0.943** |

### 依題型（預設 bge-m3 + xcloud）

| 題型 | 查詢數 | Hit@1 | Hit@3 | Hit@5 | MRR | Recall@5 |
|---|---|---|---|---|---|---|
| single（單一文件） | 153 | 0.843 | 0.961 | 0.974 | 0.900 | 0.902 |
| multi_doc（跨文件） | 21 | 0.952 | 0.952 | 0.952 | 0.952 | 0.690 |
| version_conflict（版本衝突） | 9 | 0.667 | 0.889 | 1.000 | 0.806 | 0.833 |

### 依類別（預設 bge-m3 + xcloud）

| 類別 | 查詢數 | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|
| IT支援 | 75 | 0.827 | 0.960 | 0.987 | 0.897 |
| 人事規章 | 33 | 1.000 | 1.000 | 1.000 | 1.000 |
| 專業術語 | 30 | 0.867 | 0.900 | 0.900 | 0.883 |
| 業務流程 | 24 | 0.708 | 0.917 | 0.958 | 0.807 |
| 行政財務 | 15 | 0.867 | 1.000 | 1.000 | 0.933 |
| 專案經驗 | 6 | 0.667 | 1.000 | 1.000 | 0.806 |

### 拒答門檻參考（top-1 相似度）

| 模型 | 可回答題中位數 | 可回答題最低 10% | 無答案題中位數 | 無答案題最高 |
|---|---|---|---|---|
| bge-m3 | 0.680 | 0.582 | 0.575 | 0.667 |
| xcloud | 0.558 | 0.412 | 0.408 | 0.599 |

兩者分佈重疊，**單靠相似度門檻無法可靠判斷「查不到」**，拒答需要搭配 LLM 判斷。

### 主要發現

1. 口語／錯字變體比原句難：預設設定 Hit@1 原句 0.885、變體 0.828。
2. 跨文件題的 Recall@5 偏低（0.690）：要同時找到兩份文件，前 5 名常只涵蓋其中一份。
3. 業務流程類 Hit@1 最低（0.708）。排第一的通常是**同主題的其他文件**，而不是 SOP 本身：例如「改受益人」排到 TRN-002 教材的小測驗、「解約金多久給」排到 GLO-001 術語解釋、「寄錯 email」排到 IT 資安 FAQ。正確的 SOP 多半仍在前 3 名，重排序或依文件類型加權可改善。
4. 兩個模型前 5 名都找不到的只剩縮寫與俗稱：「VIP件」、「什麼是vip案件」、「政治人物投保」（答案在 GLO-002 縮寫對照表）。

### 6.5 生成初步測試（3 題）

用 Llama-3.1-TAIDE 8B（4-bit）對 `golden.jsonl` 中已組好的 prompt 生成，並逐句比對檢索到的原文：

| 題目 | 內容正確性 | 出處標註 | 問題 |
|---|---|---|---|
| IT-01 忘記電腦密碼 | ✅ 步驟完整正確 | ❌ 沒有照 `[文件編號#章節]` 格式逐點標註 | 把「24 小時皆可使用」誤寫成「在 24 小時內」 |
| IT-05 VPN 權限申請（版本衝突題） | ✅ 正確，沒有引用已廢止的舊版內容 | 🟡 只在最後列出出處清單 | — |
| UNA-01 今年年終獎金幾個月（無答案題） | ❌ 編造「一般多在 1 至 3 個月」「約農曆年前後發放」「2024 年」 | — | 沒有直接回答「知識庫中查不到相關資訊」，最後才說無法確定 |

結論：**檢索到的內容能被正確使用，但目前的 prompt 不足以讓 8B 模型遵守引用格式與拒答規則**，需要強化 prompt 並做完整的 68 題評估。

---

## 七、待辦事項

### 優先（RAG 主流程）

- [ ] **生成答案**：新增 `rag/answer.py`，讀 `data/contexts/golden.jsonl` 用 Llama-3.1-TAIDE 8B 逐題生成並存檔（依實測約 15 tokens/秒，68 題約 15–20 分鐘）
- [ ] **強化 prompt**：初步測試中模型不照格式逐點標出處、無答案題會用常識編造，可加入正確／錯誤範例、要求先判斷資料是否足夠，或降低 temperature
- [ ] **生成品質自動評估**
  - [ ] 引用格式：回答是否含 `[文件編號#章節]`
  - [ ] 引用正確率：引用的段落是否在檢索結果中、是否對應 `expected_sources`
  - [ ] 拒答：7 題無答案題是否回答「查不到」；可回答題是否被誤拒
  - [ ] 版本衝突題是否沒有引用已廢止內容
- [ ] **人工抽查** 約 10 題的正確性與語氣
- [ ] 依結果調整 system prompt，或比較 Gemma-3-TAIDE 12B（可能需在 Colab L4／A100 上執行）
- [ ] **可互動的 RAG 對話程式**：把檢索接進 `chat.py`，回答後列出出處
- [ ] **多輪對話**：追問時（「那要怎麼申請？」）用前文補全查詢再檢索

### 檢索改善

- [ ] **查詢改寫**：用 GLO-002 縮寫對照表展開縮寫（VIP件、PEP…），或請 LLM 把口語改寫成公文用語，解決目前找不到的 3 個查詢
- [ ] **重排序（reranker）**：例如 bge-reranker-v2-m3 對前 20 名重新排序，目標提升 Hit@1
- [ ] **跨文件題**：提高 k 或在 prompt 中保留更多段落，改善 Recall@5
- [ ] **拒答機制**：相似度門檻與 LLM 判斷結合
- [ ] 評估 `xcloud-8k` 長文版

### 工程

- [ ] 生成時的 VRAM 規劃：Llama 4-bit 生成時已用到 6.8 GiB，8 GB 顯卡上**無法再同時放 bge-m3**，即時問答時 bge-m3 要改到 CPU
- [ ] Gemma 12B 在本機的替代方案：改用 GGUF 版本＋llama.cpp（可穩定地部分放 GPU、部分放 CPU），或固定在 Colab 執行
- [ ] `genai/` 改成可從 repo 根目錄直接 import（目前需要 `sys.path.insert(0, "genai")`）
- [ ] 使用者介面（例如 Streamlit），顯示答案與出處原文

---

## 八、在 Colab／Kaggle 上執行

本機 8 GB VRAM 跑 Gemma 12B 太慢時，可改到雲端。雲端的檔案在工作階段結束後會清空，每次都要重新下載模型。

```python
# 1. Colab：左側「鑰匙」圖示新增 Secret，名稱 HF_TOKEN，填 Hugging Face Read token，並開啟筆記本存取權
#    Kaggle：Add-ons → Secrets 新增 HF_TOKEN，再執行：
#    import os; from kaggle_secrets import UserSecretsClient
#    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

# 2. 取得程式並安裝套件。torch 用平台內建版本，不要裝 requirements.txt 裡的 cu128 版
!git clone https://github.com/as814788964456-ui/RAG_Carfdif.git
%cd RAG_Carfdif
!pip install -q bitsandbytes accelerate "transformers>=5"

# 3. 用已匯出的 prompt 生成（不需要 embedding 模型）
import json, sys
sys.path.insert(0, "genai")
from llama_taide import LlamaTaide
llm = LlamaTaide()
item = json.loads(open("data/contexts/golden.jsonl", encoding="utf-8").readline())
print(llm.generate(item["messages"]))
```

- T4（Colab 免費版、Kaggle）不支援 bf16，會自動改用 fp16；Gemma 在 fp16 下可能產生 NaN，要跑 Gemma 建議用 L4 或 A100。
- Kaggle 可把模型存成 Dataset，下次直接掛載不用重新下載。

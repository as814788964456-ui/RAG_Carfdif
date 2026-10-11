"""讀取知識庫、依章節切塊，並用 embedding 模型建立向量索引。

每份文件依 `##`／`###` 標題切塊，每塊帶入 front matter 的 metadata。
送進 embedding 的內容會加上「文件標題｜章節路徑」，讓短段落也有上下文。

用法（在 TAIDE 根目錄執行）:
    python -m rag.ingest                       # 預設建 bge-m3 與 xcloud 兩份索引
    python -m rag.ingest -m xcloud             # 只建其中一份
    python -m rag.ingest --chunks-only         # 只切塊，不建向量（檢查切塊結果用）
輸出:
    data/index/chunks.jsonl   所有 chunk（各模型共用）
    data/index/<模型>.npy      chunk 向量，順序與 chunks.jsonl 相同
"""
import argparse
import datetime
import json
import re
import time

import numpy as np
import yaml

from rag.common import CHUNKS_PATH, EMBEDDER_KEYS, INDEX_DIR, KB_DIR, get_embedder, vectors_path

HEADING = re.compile(r"^(#{2,3})\s+(.+?)\s*$")
META_FIELDS = [
    "doc_id", "title", "department", "category", "doc_type",
    "version", "effective_date", "status", "superseded_by", "confidentiality",
]


def parse_document(path):
    raw = path.read_text(encoding="utf-8")
    _, front_matter, body = raw.split("---", 2)
    meta = yaml.safe_load(front_matter)
    meta = {k: str(v) if isinstance(v, datetime.date) else v for k, v in meta.items()}
    return meta, body


def split_sections(body):
    """回傳 [(章節路徑, 內文)]。第一個 ## 之前的前言（H1 與虛構資料聲明）不收。"""
    sections, path, lines = [], None, []
    in_code = False

    def flush():
        text = "\n".join(lines).strip()
        if path and text:
            sections.append((list(path), text))

    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING.match(line)
        if not m:
            lines.append(line)
            continue
        flush()
        lines = []
        level, heading = m.groups()
        if level == "##":
            path = [heading]
        else:
            path = (path[:1] if path else []) + [heading]
    flush()
    return sections


def build_chunks():
    chunks = []
    for path in sorted(KB_DIR.glob("[0-9]*/*.md")):
        meta, body = parse_document(path)
        for i, (section_path, text) in enumerate(split_sections(body)):
            chunk = {"id": f"{meta['doc_id']}#{i}"}
            chunk.update({k: meta.get(k) for k in META_FIELDS})
            chunk.update({
                "section": section_path[-1],
                "path": section_path,
                "file": path.relative_to(KB_DIR).as_posix(),
                "text": text,
                "content": f"{meta['title']}｜{' > '.join(section_path)}\n{text}",
            })
            chunks.append(chunk)
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--models", nargs="+", choices=EMBEDDER_KEYS, default=["bge-m3", "xcloud"])
    parser.add_argument("--chunks-only", action="store_true")
    args = parser.parse_args()

    chunks = build_chunks()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    n_docs = len({c["doc_id"] for c in chunks})
    print(f"{n_docs} 份文件，切成 {len(chunks)} 塊 → {CHUNKS_PATH.relative_to(KB_DIR.parent.parent)}")
    if args.chunks_only:
        return

    contents = [c["content"] for c in chunks]
    for key in args.models:
        t0 = time.perf_counter()
        vecs = get_embedder(key).embed_documents(contents)
        np.save(vectors_path(key), vecs)
        print(f"[{key}] 維度 {vecs.shape[1]}，{time.perf_counter() - t0:.1f} 秒 → {vectors_path(key).name}")


if __name__ == "__main__":
    main()

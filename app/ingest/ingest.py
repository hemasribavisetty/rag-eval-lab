# app/ingest/ingest.py
import os
import json
import argparse
from pathlib import Path
from pypdf import PdfReader
from tqdm import tqdm

CHUNK_SIZE = 1000  # characters, change if you want bigger/smaller chunks

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"[WARN] Could not read PDF {file_path}: {e}")
        return ""

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=0):
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        start = end - overlap if overlap and (end - overlap) > start else end
    return chunks

def ingest(input_dir, output_file, chunk_size=CHUNK_SIZE, overlap=0):
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(input_dir.iterdir()) if p.is_file()]
    print(f"[INFO] Found {len(files)} files in {input_dir}")

    if not files:
        print("[WARN] No files to ingest. Add .txt/.md/.pdf files to the data/ folder.")
        return

    all_chunks = []
    doc_id = 0

    for file in tqdm(files, desc="Files"):
        suffix = file.suffix.lower()
        text = ""
        if suffix in [".txt", ".md"]:
            print(f"[READ] Text file: {file.name}")
            text = read_txt(file)
        elif suffix == ".pdf":
            print(f"[READ] PDF file: {file.name}")
            text = read_pdf(file)
        else:
            print(f"[SKIP] Unsupported file type: {file.name}")
            continue

        if not text or text.strip()=="":
            print(f"[WARN] No text extracted from {file.name}")
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        print(f"[INFO] {file.name} -> {len(chunks)} chunks")

        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "doc_id": doc_id,
                "chunk_id": idx,
                "source_file": file.name,
                "text": chunk
            })
        doc_id += 1

    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"[DONE] Saved {len(all_chunks)} chunks to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data", help="Input folder with docs")
    parser.add_argument("--output_file", default="experiments/chunks.jsonl", help="Output JSONL chunks file")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=0, help="Chunk overlap in characters")
    args = parser.parse_args()

    ingest(args.input_dir, args.output_file, chunk_size=args.chunk_size, overlap=args.overlap)

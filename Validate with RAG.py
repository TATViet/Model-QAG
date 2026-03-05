import os
import re
import json
import csv
import time
import random
import argparse
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    set_seed,
)
import evaluate

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ===================== CONFIG =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COLLECTION_NAME = "my_collection"

INSTRUCTIONS = [
    "Đặt ra một số câu hỏi và câu trả lời cho đoạn văn sau.",
    "Tạo ra một vài cặp câu hỏi và câu trả lời tương ứng với đoạn văn sau.",
    "Tạo ra một số câu hỏi và câu trả lời của chúng dựa trên văn bản sau.",
    "Xây dựng câu hỏi và câu trả lời từ đoạn văn đã cho.",
    "Phát triển một tập hợp cặp câu hỏi-câu trả lời cho văn bản bên dưới.",
    "Xây dựng câu hỏi cùng với câu trả lời của chúng cho nội dung sau.",
    "Sản xuất cặp câu hỏi-câu trả lời được lấy từ đoạn văn.",
    "Nghĩ ra một số câu hỏi và câu trả lời liên quan đến đoạn văn sau.",
    "Xây dựng danh sách câu hỏi và câu trả lời cho văn bản đã cho.",
    "Tạo ra các kết hợp câu hỏi-câu trả lời dựa trên đoạn văn được cung cấp.",
    "Tạo ra các cặp QA cho đoạn văn sau.",
    "Xây dựng một số Q&A từ văn bản bên dưới.",
    "Phát triển câu hỏi và câu trả lời tương ứng với đoạn văn.",
]

# -----------------------------
# Helpers
# -----------------------------
def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def contains_answer_in_context(answer: str, context: str) -> int:
    if not answer:
        return 0
    ans = normalize_text(answer).lower()
    ctx = normalize_text(context).lower()
    return 1 if ans in ctx else 0

QA_PARSE_RE = re.compile(
    r"question\s*:\s*(?P<q>.*?)(?:\s+answer\s*:\s*(?P<a>.*))?$",
    flags=re.IGNORECASE | re.DOTALL
)

def parse_qa_from_text(gen_text: str) -> Tuple[str, str]:
    """
    Parse string formatted like:
      "question: ... answer: ..."
    Return (question, answer). If fail: (whole, "").
    """
    t = normalize_text(gen_text)
    m = QA_PARSE_RE.match(t)
    if not m:
        return (t, "")
    q = normalize_text(m.group("q") or "")
    a = normalize_text(m.group("a") or "")
    return (q, a)

# -----------------------------
# RAG Functions
# -----------------------------
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=DEVICE)
client = QdrantClient(host="localhost", port=6333)

def advanced_retrieve(keyword, top_k=4):
    query_embedding = embedder.encode(keyword).tolist()
    
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k * 5,
        with_payload=True
    ).points

    if not search_result:
        return []

    chunks = [hit.payload.get('text', '') for hit in search_result]
    headings = [hit.payload.get('heading', '') for hit in search_result]
    scores = torch.tensor([hit.score for hit in search_result])

    max_score = scores.max().item()
    threshold = max_score * 0.70
    fallback_threshold = max_score * 0.50

    filtered_idx = torch.nonzero(scores >= threshold).squeeze()
    if filtered_idx.numel() == 0:
        filtered_idx = torch.nonzero(scores >= fallback_threshold).squeeze()
    if filtered_idx.numel() == 0:
        filtered_idx = torch.topk(scores, top_k * 2).indices

    filtered_idx = filtered_idx[torch.argsort(scores[filtered_idx], descending=True)]

    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    top_indices = []
    for idx in filtered_idx:
        idx = idx.item()
        if pattern.search(chunks[idx]) or pattern.search(headings[idx]):
            top_indices.append(idx)
            if len(top_indices) >= top_k:
                break

    if len(top_indices) < top_k:
        for idx in filtered_idx:
            idx = idx.item()
            if idx not in top_indices:
                top_indices.append(idx)
            if len(top_indices) >= top_k:
                break

    selected_chunks = [chunks[i] for i in top_indices]
    return selected_chunks

# -----------------------------
# Evaluation
# -----------------------------
@dataclass
class EvalConfig:
    method: str
    max_gen_len: int
    num_beams: int
    instr_mode: str
    instr_fixed_id: int
    seed: int

def compute_text_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    preds = [normalize_text(x) for x in preds]
    refs = [normalize_text(x) for x in refs]

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    chrf = evaluate.load("chrf")

    rouge_res = rouge.compute(predictions=preds, references=refs, use_stemmer=False)
    bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs])
    chrf_res = chrf.compute(predictions=preds, references=refs)
    precisions = bleu_res.get("precisions", [float("nan")] * 4)
    if not isinstance(precisions, list):
        precisions = [float("nan")] * 4
    precisions = list(precisions[:4]) + [float("nan")] * max(0, 4 - len(precisions))

    return {
        "rouge1": float(rouge_res.get("rouge1", float("nan"))),
        "rouge2": float(rouge_res.get("rouge2", float("nan"))),
        "rougeL": float(rouge_res["rougeL"]),
        "bleu": float(bleu_res["bleu"]),
        "bleu_p1": float(precisions[0]),
        "bleu_p2": float(precisions[1]),
        "bleu_p3": float(precisions[2]),
        "bleu_p4": float(precisions[3]),
        "bleu_bp": float(bleu_res.get("brevity_penalty", float("nan"))),
        "bleu_len_ratio": float(bleu_res.get("length_ratio", float("nan"))),
        "bleu_pred_len": float(bleu_res.get("translation_length", float("nan"))),
        "bleu_ref_len": float(bleu_res.get("reference_length", float("nan"))),
        "chrf": float(chrf_res["score"]),
    }


def prefix_metrics(prefix: str, m: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in m.items()}


def empty_prefixed_metrics(prefix: str) -> Dict[str, float]:
    keys = [
        "rouge1",
        "rouge2",
        "rougeL",
        "bleu",
        "bleu_p1",
        "bleu_p2",
        "bleu_p3",
        "bleu_p4",
        "bleu_bp",
        "bleu_len_ratio",
        "bleu_pred_len",
        "bleu_ref_len",
        "chrf",
    ]
    return {f"{prefix}_{k}": float("nan") for k in keys}

def evaluate_with_rag(model, tokenizer, raw_val, eval_cfg: EvalConfig) -> Dict[str, float]:
    preds = []
    rng = random.Random(eval_cfg.seed)

    def pick_instruction() -> str:
        if eval_cfg.instr_mode == "fixed":
            idx = max(0, min(eval_cfg.instr_fixed_id, len(INSTRUCTIONS) - 1))
            return INSTRUCTIONS[idx]
        return rng.choice(INSTRUCTIONS)

    for sample in raw_val:
        context = sample["context"]
        answer = sample["answers"]["text"][0]
        chunks = advanced_retrieve(answer)
        enhanced_context = context + " " + " ".join(chunks)

        instruction = pick_instruction()
        input_text = f"generate answer: {instruction} {enhanced_context}"

        inputs = tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=eval_cfg.max_gen_len,
                num_beams=eval_cfg.num_beams,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                no_repeat_ngram_size=3
            )
        
        qa = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        preds.append(qa)

    gold_q = [normalize_text(q) for q in raw_val["question"]]
    gold_a = [normalize_text(a["text"][0]) for a in raw_val["answers"]]
    contexts = [sample["context"] for sample in raw_val]  # Use original context for a_in_ctx

    parsed_q, parsed_a, format_ok = [], [], []
    for t in preds:
        q, a = parse_qa_from_text(t)
        parsed_q.append(q)
        parsed_a.append(a)
        format_ok.append(1 if q else 0)

    n = min(len(parsed_q), len(gold_q), len(gold_a), len(contexts))
    parsed_q, parsed_a = parsed_q[:n], parsed_a[:n]
    gold_q, gold_a = gold_q[:n], gold_a[:n]
    contexts = contexts[:n]
    format_ok = format_ok[:n]

    qm = compute_text_metrics(parsed_q, gold_q)
    am = compute_text_metrics(parsed_a, gold_a)

    a_in_ctx = [contains_answer_in_context(pa, ctx) for pa, ctx in zip(parsed_a, contexts)]

    return {
        **prefix_metrics("q", qm),
        **prefix_metrics("a", am),
        "a_in_ctx": float(np.mean(a_in_ctx)) if len(a_in_ctx) else 0.0,
        "qa_format_ok": float(np.mean(format_ok)) if len(format_ok) else 0.0,
    }

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="taidng/UIT-ViQuAD2.0")
    parser.add_argument("--model_name", type=str, default="bartpho_syllable_multitask_qag_checkpoint/checkpoint-14430")
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--method", type=str, default="instruction", choices=["pipeline", "multitask", "end2end", "instruction"])
    parser.add_argument("--instr_mode", type=str, default="random", choices=["na", "fixed", "random"])
    parser.add_argument("--instr_fixed_id", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--max_src_len", type=int, default=512)
    parser.add_argument("--max_tgt_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_gen_len", type=int, default=128)

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    set_seed(args.seed)

    dataset = load_dataset(args.dataset_name)
    dataset["validation"] = dataset["validation"].filter(lambda x: not x["is_impossible"])
    raw_val = dataset["validation"]

    tok_name = args.tokenizer_name or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(DEVICE)
    model.eval()

    eval_cfg = EvalConfig(
        method=args.method,
        max_gen_len=args.max_gen_len,
        num_beams=args.num_beams,
        instr_mode=args.instr_mode,
        instr_fixed_id=args.instr_fixed_id,
        seed=args.seed
    )
    metrics = evaluate_with_rag(model, tokenizer, raw_val, eval_cfg)

    row = {
        "model_name": args.model_name,
        "tokenizer_name": tok_name,
        "method": args.method,
        "instr_mode": args.instr_mode,
        "instr_fixed_id": args.instr_fixed_id if (args.method == "instruction" and args.instr_mode == "fixed") else -1,
        "num_beams": args.num_beams,
        "max_gen_len": args.max_gen_len,
        "max_src_len": args.max_src_len,
        "max_tgt_len": args.max_tgt_len,
        **metrics,
    }

    # JSONL append
    jsonl_path = os.path.join(args.results_dir, "metrics_rag.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # CSV append
    csv_path = os.path.join(args.results_dir, "metrics_rag.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print("Saved results to:", csv_path, "and", jsonl_path)
    print("Row:", row)


if __name__ == "__main__":
    main()

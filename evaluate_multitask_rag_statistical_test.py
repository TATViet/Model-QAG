#!/usr/bin/env python3
import os
import re
import time
import random
import subprocess
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from scipy import stats
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import bert_score
from tqdm import tqdm

# ====================== SUPER AUTO INSTALL ======================
def _ensure_all_packages():
    required = {
        "evaluate": None,
        "rouge_score": None,
        "sacrebleu": None,
        "qdrant-client": "1.17.1",
        "bert-score": None,
        "datasets": None,
        "transformers": None,
        "sentence-transformers": None,
    }
    
    print("🔄 Đang kiểm tra và cài đặt tất cả thư viện cần thiết...")
    to_install = []
    
    for pkg, min_ver in required.items():
        try:
            installed = version(pkg.replace("-", "_"))
            if min_ver and installed < min_ver:
                print(f"   ↳ {pkg} cũ ({installed}) → upgrade")
                to_install.append(pkg)
            else:
                print(f"   ✓ {pkg} {installed}")
        except PackageNotFoundError:
            print(f"   ↳ {pkg} chưa có → cài mới")
            to_install.append(pkg)
    
    if to_install:
        print(f"   Đang cài/upgrade {len(to_install)} package...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            *to_install, "--upgrade", "--quiet", "--no-warn-script-location"
        ])
        print("✅ Đã cài/upgrade xong tất cả thư viện!")
    else:
        print("✅ Tất cả thư viện đã mới nhất!")

_ensure_all_packages()

# Import sau khi chắc chắn đã có
import evaluate
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams
# ============================================================

# ====================== CONFIG ======================
SCRIPT_DIR = Path(__file__).parent.absolute()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_PATH = "multitask_rag_evaluation_results.csv"

# ================== THAY ĐỔI Ở ĐÂY ==================
TEST_SIZE = 300
EVALUATE_UNANSWERABLE = False
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "my_collection"
RAG_TOP_K = 3
# ====================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Qdrant: {QDRANT_HOST}:{QDRANT_PORT} | Collection: {COLLECTION_NAME}")
print(f"[INFO] Mode: {'UNANSWERABLE (khó)' if EVALUATE_UNANSWERABLE else 'ANSWERABLE'}")

# ====================== LOAD MODELS ======================
def load_model(model_path: str, name: str):
    print(f"[INFO] Loading {name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
    model.eval()
    return tokenizer, model

vit5_tokenizer, vit5_model = load_model(
    os.path.join(SCRIPT_DIR, "..", "PythonModels", "vit5_base_multitask_qag_checkpoint", "checkpoint-14430"),
    "ViT5-base"
)
bart_tokenizer, bart_model = load_model(
    os.path.join(SCRIPT_DIR, "..", "PythonModels", "bartpho_syllable_multitask_qag_checkpoint", "checkpoint-14430"),
    "BARTpho-syllable"
)

rag_embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ====================== REAL QDRANT RAG (API MỚI 2026) ======================
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
print(f"[DEBUG] qdrant-client version: {version('qdrant-client')} | Server OK")

def get_rag_output(context: str) -> str:
    """Context → encode → Qdrant query_points (API mới) → OutputRAG"""
    query_embedding = rag_embedder.encode(context, convert_to_tensor=False).tolist()
    
    # ✅ API CHUẨN 2026: query_points + query=vector
    response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,                    # vector trực tiếp
        limit=RAG_TOP_K,
        search_params=SearchParams(
            hnsw_ef=128,
            exact=False
        )
    )
    
    # response.points là list các hit
    retrieved_texts = [hit.payload.get("text", hit.payload.get("content", str(hit.payload)))
                       for hit in response.points]
    return "\n\n".join(retrieved_texts) if retrieved_texts else "[No relevant chunk found]"


# ====================== METRICS + ERROR PROPAGATION ======================
def compute_metrics(preds: list, refs: list, contexts: list, gen_answers: list, gold_answers: list):
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    chrf = evaluate.load("chrf")
    
    rouge_res = rouge.compute(predictions=preds, references=refs)
    bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs])
    chrf_res = chrf.compute(predictions=preds, references=refs)
    
    P, R, F1 = bert_score.score(preds, refs, lang="vi", verbose=False)
    bert_f1 = F1.mean().item()
    
    answer_exact = [1 if normalize_text(ga) == normalize_text(gold) else 0 
                    for ga, gold in zip(gen_answers, gold_answers)]
    answer_rouge = rouge.compute(predictions=gen_answers, references=gold_answers)["rougeL"]
    
    cosine_sim = []
    for p, r in zip(preds, refs):
        emb_p = rag_embedder.encode(p)
        emb_r = rag_embedder.encode(r)
        sim = float(np.dot(emb_p, emb_r) / (np.linalg.norm(emb_p) * np.linalg.norm(emb_r)))
        cosine_sim.append(sim)
    
    a_in_ctx = [1 if normalize_text(a) in normalize_text(c) else 0 for a, c in zip(refs, contexts)]
    
    return {
        "rougeL": round(rouge_res["rougeL"], 4),
        "bleu": round(bleu_res["bleu"], 4),
        "chrf": round(chrf_res["score"], 4),
        "bertscore_f1": round(bert_f1, 4),
        "cosine_sim": round(np.mean(cosine_sim), 4),
        "a_in_ctx_ratio": round(np.mean(a_in_ctx), 4),
        "answer_exact_match": round(np.mean(answer_exact), 4),
        "answer_rougeL": round(answer_rouge, 4),
    }

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

# ====================== INFERENCE ======================
def generate_qa(model, tokenizer, context: str, use_rag: bool = False):
    start = time.time()
    
    if use_rag:
        output_rag = get_rag_output(context)
        aug_context = output_rag + "\n\n" + context
    else:
        aug_context = context
    
    ag_input = "generate answer: " + aug_context
    inputs = tokenizer(ag_input, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
    ag_out = model.generate(**inputs, max_length=128, num_beams=8, early_stopping=True)
    answer = tokenizer.decode(ag_out[0], skip_special_tokens=True).strip()
    
    qg_input = f"generate question: {aug_context} answer: {answer}"
    inputs = tokenizer(qg_input, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
    qg_out = model.generate(**inputs, max_length=128, num_beams=8, early_stopping=True)
    question = tokenizer.decode(qg_out[0], skip_special_tokens=True).strip()
    
    inf_time = time.time() - start
    return question, answer, inf_time

# ====================== MAIN + RESUME ======================
dataset = load_dataset("taidng/UIT-ViQuAD2.0")
filter_condition = lambda x: x["is_impossible"] if EVALUATE_UNANSWERABLE else not x["is_impossible"]
test_set = dataset["validation"].filter(filter_condition).shuffle(seed=SEED).select(range(TEST_SIZE))

results = []
processed = set()
if os.path.exists(CSV_PATH):
    print(f"[INFO] Resume từ file {CSV_PATH}")
    df_existing = pd.read_csv(CSV_PATH)
    results = df_existing.to_dict('records')
    processed = {(row["model"], row["rag"], row["sample_id"]) for row in results}

print(f"[INFO] Bắt đầu evaluation ({len(test_set)} mẫu - {'Unanswerable' if EVALUATE_UNANSWERABLE else 'Answerable'})...")

for model_name, model, tokenizer in [("ViT5-base", vit5_model, vit5_tokenizer), 
                                     ("BARTpho-syllable", bart_model, bart_tokenizer)]:
    for use_rag in [False, True]:
        rag_str = "RAG" if use_rag else "non-RAG"
        print(f"   → {model_name} | {rag_str}")
        
        for idx, ex in enumerate(tqdm(test_set, desc=f"   Evaluating {model_name} {rag_str}", unit="sample")):
            key = (model_name, rag_str, idx)
            if key in processed:
                continue
            
            context = ex["context"]
            gold_q = ex["question"]
            gold_a = ex["answers"]["text"][0] if ex["answers"]["text"] else ""
            
            q, a, inf_time = generate_qa(model, tokenizer, context, use_rag)
            
            metrics = compute_metrics([q], [gold_q], [context], [a], [gold_a])
            
            row = {
                "model": model_name,
                "rag": rag_str,
                "sample_id": idx,
                "is_impossible": ex["is_impossible"],
                "gold_question": gold_q,
                "gold_answer": gold_a,
                "gen_question": q,
                "gen_answer": a,
                "inf_time_sec": round(inf_time, 4),
                **metrics
            }
            results.append(row)
            
            if len(results) % 20 == 0:
                pd.DataFrame(results).to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
                print(f"   💾 Đã lưu checkpoint ({len(results)} mẫu)")

pd.DataFrame(results).to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

# ====================== ERROR PROPAGATION ANALYSIS ======================
print("\n" + "="*90)
print("KẾT QUẢ + PHÂN TÍCH ERROR PROPAGATION")
print("="*90)

df = pd.DataFrame(results)

for model_name in ["ViT5-base", "BARTpho-syllable"]:
    sub = df[df["model"] == model_name]
    if len(sub) == 0: continue
    delta = sub[sub["rag"] == "RAG"]["rougeL"].mean() - sub[sub["rag"] == "non-RAG"]["rougeL"].mean()
    print(f"{model_name:15} RAG gain (ROUGE-L): {delta:.4f}")

print("\n--- Error Propagation Analysis (khi answer sai → question giảm bao nhiêu?) ---")
for model_name in ["ViT5-base", "BARTpho-syllable"]:
    sub = df[df["model"] == model_name]
    correct = sub[sub["answer_exact_match"] == 1]
    wrong   = sub[sub["answer_exact_match"] < 1]
    if len(correct) == 0 or len(wrong) == 0: continue
    drop = correct["rougeL"].mean() - wrong["rougeL"].mean()
    print(f"{model_name:15} Question ROUGE-L drop khi answer sai: {drop:.4f} "
          f"({len(wrong)}/{len(sub)} mẫu)")

print(f"\n✅ XONG! Kết quả lưu tại: {CSV_PATH}")
import os
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from qdrant_client import QdrantClient
from datasets import load_dataset

# ====================== CONFIG ======================
CHUNK_SIZE = 420
NUM_BEAMS = 8
SIMILARITY_THRESHOLD = 0.90 
PAIR_QUALITY_THRESHOLD = 0.60 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_OUTPUT = "chunking_vs_rag_chunking_results.csv"
TEST_SIZE = 300 # Số lượng mẫu chạy cho bài báo
RAG_TOP_K = 3

# ====================== LOAD MODELS ======================
SCRIPT_DIR = Path(__file__).parent.absolute()
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "PythonModels", "vit5_base_multitask_qag_checkpoint", "checkpoint-14430")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
rag_embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
qdrant_client = QdrantClient(host="localhost", port=6333)

# ====================== CORE PIPELINE ======================

def process_and_generate(text_to_chunk):
    """Chia chunk và sinh bộ QA chất lượng từ một đoạn text bất kỳ"""
    chunks = [text_to_chunk[i:i+CHUNK_SIZE] for i in range(0, len(text_to_chunk), CHUNK_SIZE - 100)]
    valid_pairs = []
    
    for chunk in chunks:
        if len(chunk.strip()) < 100: continue
        
        # Sinh QA
        in_a = tokenizer(f"generate answer: {chunk}", return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
        out_a = model.generate(**in_a, max_length=128, num_beams=NUM_BEAMS)
        ans = tokenizer.decode(out_a[0], skip_special_tokens=True).strip()
        
        score = util.cos_sim(rag_embedder.encode(ans), rag_embedder.encode(chunk)).item()
        if score < PAIR_QUALITY_THRESHOLD: continue
        
        in_q = tokenizer(f"generate question: {chunk} answer: {ans}", return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
        out_q = model.generate(**in_q, max_length=128, num_beams=NUM_BEAMS)
        ques = tokenizer.decode(out_q[0], skip_special_tokens=True).strip()
        
        # Similarity Filter
        is_duplicate = False
        if valid_pairs:
            q_emb = rag_embedder.encode(ques, convert_to_tensor=True)
            prev_qs_emb = rag_embedder.encode([p['q'] for p in valid_pairs], convert_to_tensor=True)
            if torch.max(util.cos_sim(q_emb, prev_qs_emb)).item() > SIMILARITY_THRESHOLD:
                is_duplicate = True
        
        if not is_duplicate:
            valid_pairs.append({'q': ques, 'a': ans, 's': score})
            
    return valid_pairs

def get_retrieval(context):
    query_vector = rag_embedder.encode(context).tolist()
    response = qdrant_client.query_points(collection_name="my_collection", query=query_vector, limit=RAG_TOP_K)
    return "\n\n".join([hit.payload.get("text", "") for hit in response.points])

# ====================== EXECUTION ======================
dataset = load_dataset("taidng/UIT-ViQuAD2.0", split="validation").shuffle(seed=42).select(range(TEST_SIZE))
final_results = []

for idx, ex in enumerate(tqdm(dataset, desc="So sánh Chunking vs RAG-Chunking")):
    ctx = ex["context"]
    
    # 1. PHƯƠNG PHÁP A: Chỉ Chunking trên Context gốc (Non-RAG + Chunking)
    non_rag_chunked_qa = process_and_generate(ctx)
    
    # 2. PHƯƠNG PHÁP B: Chunking trên Context + Retrieval (RAG + Chunking)
    retrieved = get_retrieval(ctx)
    rag_chunked_qa = process_and_generate(retrieved + "\n\n" + ctx)
    
    # Lưu kết quả Phương pháp A
    for i, p in enumerate(non_rag_chunked_qa):
        final_results.append({
            "topic_id": idx,
            "method": "Chunking_Only",
            "qa_index": i + 1,
            "question": p['q'],
            "answer": p['a'],
            "quality_score": round(p['s'], 4)
        })
        
    # Lưu kết quả Phương pháp B
    for i, p in enumerate(rag_chunked_qa):
        final_results.append({
            "topic_id": idx,
            "method": "RAG_Chunking",
            "qa_index": i + 1,
            "question": p['q'],
            "answer": p['a'],
            "quality_score": round(p['s'], 4)
        })

# Xuất kết quả
df = pd.DataFrame(final_results)
df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")

# Thống kê nhanh
print("\n" + "="*50)
print("THỐNG KÊ KẾT QUẢ")
summary = df.groupby(['method', 'topic_id']).size().reset_index(name='counts').groupby('method')['counts'].mean()
print(f"Số lượng QA trung bình/Topic (Chunking Only): {summary.get('Chunking_Only', 0):.2f}")
print(f"Số lượng QA trung bình/Topic (RAG Chunking):  {summary.get('RAG_Chunking', 0):.2f}")
print(f"Chất lượng trung bình (Quality Score): \n{df.groupby('method')['quality_score'].mean()}")
print("="*50)
import os
import json
import re
import string
import unicodedata
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, EncoderDecoderModel
import torch
import nltk
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

nltk.download('wordnet')
nltk.download('omw-1.4')

def normalize_answer(s):
    """Standard SQuAD-style normalization for Vietnamese (adapted)"""
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def lower(text):
        return text.lower()
    
    # For Vietnamese, no articles to remove, but normalize unicode
    s = unicodedata.normalize('NFC', s)
    return white_space_fix(remove_punc(lower(s)))

def compute_em(pred, gold):
    return 1 if normalize_answer(pred) == normalize_answer(gold) else 0

def compute_f1(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return int(pred_tokens == gold_tokens)
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

def compute_q_metrics(gen_q, ref_q):
    if not gen_q or not ref_q:
        return {"q_bleu": 0, "q_rouge": 0, "q_meteor": 0}
    
    bleu = sentence_bleu([ref_q.split()], gen_q.split(), weights=(0.25, 0.25, 0.25, 0.25))
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge = scorer.score(ref_q, gen_q)['rougeL'].fmeasure
    meteor = meteor_score([ref_q.split()], gen_q.split())
    return {"q_bleu": bleu, "q_rouge": rouge, "q_meteor": meteor}

# Load dataset
dataset = load_dataset("taidng/UIT-ViQuAD2.0")['test']
dataset = dataset.filter(lambda x: not x['is_impossible'])

device = "cuda" if torch.cuda.is_available() else "cpu"

# List all _qag_checkpoint directories
model_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.endswith('_qag_checkpoint')]

for model_dir in model_dirs:
    # Find last checkpoint
    checkpoints = [c for c in os.listdir(model_dir) if c.startswith('checkpoint-')]
    if not checkpoints:
        print(f"No checkpoints found in {model_dir}")
        continue
    last_ckpt = max(checkpoints, key=lambda x: int(x.split('-')[1]))
    ckpt_path = os.path.join(model_dir, last_ckpt)
    
    print(f"Evaluating {model_dir} using {last_ckpt}")
    
    # Load model and tokenizer
    try:
        if 'mbert' in model_dir or 'xlmr' in model_dir:
            model = EncoderDecoderModel.from_pretrained(ckpt_path)
            tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path)
            tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model from {ckpt_path}: {e}")
        continue
    
    # Determine method from dir name
    if 'instruction' in model_dir:
        method = 'instruction'
        instruction = "Tạo ra một vài cặp câu hỏi và câu trả lời tương ứng với đoạn văn sau."
    elif 'pipeline' in model_dir or 'multitask' in model_dir:
        method = 'pipeline'
    else:
        method = 'qag'
    
    # Initialize metrics sum
    metrics_sum = {"q_bleu": 0, "q_rouge": 0, "q_meteor": 0, "a_em": 0, "a_f1": 0}
    count = 0
    
    for example in dataset:
        context = example['context']
        ref_q = example['question']
        answers = example.get('answers', None)
        if answers is None or not answers.get('text', []):
            print(f"Skipping example with id {example.get('id', 'unknown')} due to missing answers")
            continue
        ref_a = answers['text'][0]
        
        gen_q = ""
        gen_a = ""
        
        try:
            if method == 'pipeline':
                # Generate answer first
                ag_input = "generate answer: " + context
                inputs = tokenizer(ag_input, return_tensors="pt", max_length=512, truncation=True).to(device)
                ag_ids = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=128, num_beams=4, early_stopping=True)
                gen_a = tokenizer.decode(ag_ids[0], skip_special_tokens=True)
                
                # Then generate question
                qg_input = "generate question: " + context + " answer: " + gen_a
                inputs = tokenizer(qg_input, return_tensors="pt", max_length=512, truncation=True).to(device)
                qg_ids = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=128, num_beams=4, early_stopping=True)
                gen_q = tokenizer.decode(qg_ids[0], skip_special_tokens=True)
            
            else:
                # For instruction or qag
                if method == 'instruction':
                    input_text = instruction + " " + context
                else:
                    input_text = "generate qa: " + context
                
                inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
                output_ids = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=128, num_beams=4, early_stopping=True)
                output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # Parse output assuming "question: <q> answer: <a>"
                if "question:" in output and "answer:" in output:
                    parts = re.split(r'(question:|answer:)', output)
                    q_idx = parts.index('question:') + 1 if 'question:' in parts else None
                    a_idx = parts.index('answer:') + 1 if 'answer:' in parts else None
                    if q_idx and a_idx:
                        gen_q = parts[q_idx].strip()
                        gen_a = ''.join(parts[a_idx:]).strip()  # In case multiple
                else:
                    print(f"Parse failed for output: {output}")
        
            if gen_q and gen_a:
                q_m = compute_q_metrics(gen_q, ref_q)
                a_em = compute_em(gen_a, ref_a)
                a_f1 = compute_f1(gen_a, ref_a)
                
                metrics_sum["q_bleu"] += q_m["q_bleu"]
                metrics_sum["q_rouge"] += q_m["q_rouge"]
                metrics_sum["q_meteor"] += q_m["q_meteor"]
                metrics_sum["a_em"] += a_em
                metrics_sum["a_f1"] += a_f1
                count += 1
        
        except Exception as e:
            print(f"Error during generation for {model_dir}: {e}")
    
    if count > 0:
        avg_metrics = {k: v / count for k, v in metrics_sum.items()}
        output_dir = "validating"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, model_dir.replace("_qag_checkpoint", "") + "_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(avg_metrics, f, indent=4, ensure_ascii=False)
        print(f"Results saved to {output_file}")
    else:
        print(f"No valid generations for {model_dir}")
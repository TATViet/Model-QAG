import os
import random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint

# Load the dataset
dataset = load_dataset("taidng/UIT-ViQuAD2.0")

# Filter to only answerable questions (is_impossible == False)
dataset['train'] = dataset['train'].filter(lambda x: not x['is_impossible'])
dataset['validation'] = dataset['validation'].filter(lambda x: not x['is_impossible'])

# List of instructions based on Table 1 from the paper (translated to English for simplicity; adjust if needed)
instructions = [
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
    "Phát triển câu hỏi và câu trả lời tương ứng với đoạn văn."
]

# Define preprocess functions for each method
def preprocess_pipeline(examples, tokenizer):
    # For Pipeline: Train a Question Generation (QG) model
    # Input: "generate question: <context> answer: <a>"
    # Output: "<q>"
    inputs = ["generate question: " + context + " answer: " + a['text'][0] for context, a in zip(examples['context'], examples['answers'])]
    targets = examples['question']
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_multitask(examples, tokenizer):
    # For Multitask: Mix AG and QG tasks
    # Duplicate examples: half for AG, half for QG
    # AG: Input "generate answer: <context>" -> "<a>"
    # QG: Input "generate question: <context> answer: <a>" -> "<q>"
    # Note: In practice, we create two entries per example
    inputs = []
    targets = []
    for context, q, a in zip(examples['context'], examples['question'], examples['answers']):
        # AG sample
        inputs.append("generate answer: " + context)
        targets.append(a['text'][0])
        # QG sample
        inputs.append("generate question: " + context + " answer: " + a['text'][0])
        targets.append(q)
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_end2end(examples, tokenizer):
    # For End2End (Seq2Seq): Similar to your original code
    inputs = ["generate qa: " + context for context in examples['context']]
    targets = ["question: " + q + " answer: " + a['text'][0] for q, a in zip(examples['question'], examples['answers'])]
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_instruction(examples, tokenizer):
    # For Instruction: Randomly select an instruction + context
    # Input: "<random_instruction> <context>"
    # Output: "question: <q> answer: <a>"
    inputs = [random.choice(instructions) + " " + context for context in examples['context']]
    targets = ["question: " + q + " answer: " + a['text'][0] for q, a in zip(examples['question'], examples['answers'])]
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Mapping of methods to preprocess functions
method_preprocess = {
    "pipeline": preprocess_pipeline,
    "multitask": preprocess_multitask,
    "end2end": preprocess_end2end,
    "instruction": preprocess_instruction
}

# List of models (model_name, tokenizer_name)
models = [
    ("google/mt5-small", "google/mt5-small"),
    ("vinai/bartpho-syllable", "vinai/bartpho-syllable"),
    ("VietAI/vit5-base", "VietAI/vit5-base")
]

# Methods to train (now including end2end)
methods = ["pipeline", "multitask", "end2end", "instruction"]

def train_model(model_name, tokenizer_name, output_dir, preprocess_func):
    # Check for existing checkpoint
    last_checkpoint = get_last_checkpoint(output_dir) if os.path.exists(output_dir) else None
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Apply preprocess (batched=True for efficiency)
    tokenized_train = dataset['train'].map(lambda x: preprocess_func(x, tokenizer), batched=True)
    tokenized_val = dataset['validation'].map(lambda x: preprocess_func(x, tokenizer), batched=True)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True,  # Assuming GPU available
        save_strategy="steps",
        save_steps=1000,  # Save checkpoint every 1000 steps
        resume_from_checkpoint=last_checkpoint  # Resume if checkpoint exists
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
    )
    
    trainer.train(resume_from_checkpoint=last_checkpoint is not None)

# Train all 12 combinations
for method in methods:
    for model_name, tokenizer_name in models:
        # Create unique output_dir
        model_short = model_name.split('/')[-1].replace('-', '_')  # e.g., "mt5_small"
        if method == "end2end":
            output_dir = f"{model_short}_qag_checkpoint"
        else:
            output_dir = f"{model_short}_{method}_qag_checkpoint"
        
        preprocess_func = method_preprocess[method]
        
        print(f"Training {model_short} with {method} method...")
        try:
            train_model(model_name, tokenizer_name, output_dir, preprocess_func)
        except Exception as e:
            print(f"Error during training {model_short} {method}: {e}")
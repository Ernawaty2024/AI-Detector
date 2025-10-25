import os
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM, OVModelForSequenceClassification

# ✅ Model Directories
GPT_MODEL_NAME = "gpt2"
GPT_MODEL_DIR = "optimized_detectgpt_openvino"
ROBERTA_MODEL_NAME = "roberta-base-openai-detector"
ROBERTA_MODEL_DIR = "optimized_roberta_openvino"

# --------------------------- #
# ✅ Load Tokenizers & Models  #
# --------------------------- #
try:
    tokenizer_gpt = AutoTokenizer.from_pretrained(GPT_MODEL_NAME)
    tokenizer_gpt.pad_token = tokenizer_gpt.eos_token
    
    tokenizer_roberta = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    tokenizer_roberta.pad_token = tokenizer_roberta.eos_token
    
    ov_gpt_model = OVModelForCausalLM.from_pretrained(GPT_MODEL_DIR)
    ov_roberta_model = OVModelForSequenceClassification.from_pretrained(ROBERTA_MODEL_DIR)
    
    print("✅ Tokenizers and Models Loaded Successfully")
except Exception as e:
    print(f"❌ Error Loading Models: {e}")
    exit(1)

# ------------------------------- #
# ✅ AI Detection Functions       #
# ------------------------------- #
def compute_log_probability_openvino(text):
    """Computes log probability using OpenVINO-optimized GPT-2 (DetectGPT)."""
    try:
        inputs = tokenizer_gpt(text, return_tensors="np", truncation=True, padding="max_length", max_length=256)
        inputs_numpy = {k: v.astype(np.int64) for k, v in inputs.items()}
        inputs_numpy["position_ids"] = np.arange(inputs_numpy["input_ids"].shape[1], dtype=np.int64).reshape(1, -1)
        beam_idx = np.array([0], dtype=np.int32)
        
        with torch.no_grad():
            output = ov_gpt_model(
                input_ids=inputs_numpy["input_ids"],
                attention_mask=inputs_numpy["attention_mask"],
                position_ids=inputs_numpy["position_ids"],
                beam_idx=beam_idx
            )
        
        logits = output.logits
        input_ids = inputs["input_ids"][0]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs[0, torch.arange(len(input_ids)), input_ids]

        return token_log_probs.sum().item()
    except Exception as e:
        print(f"❌ DetectGPT Inference Failed: {e}")
        return None


def detect_ai_openvino_roberta(text):
    """Predicts AI probability using OpenVINO-optimized RoBERTa."""
    try:
        inputs = tokenizer_roberta(text, return_tensors="np", truncation=True, padding="max_length", max_length=512)
        input_ids = np.array(inputs["input_ids"], dtype=np.int64)
        attention_mask = np.array(inputs["attention_mask"], dtype=np.int64)
        
        with torch.no_grad():
            outputs = ov_roberta_model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
        return {"human_prob": probs[0][0].item(), "ai_prob": probs[0][1].item()}
    except Exception as e:
        print(f"❌ RoBERTa Inference Failed: {e}")
        return None

# ------------------------------- #
# ✅ Dataset Processing Functions  #
# ------------------------------- #
CHUNK_SIZE = 5000

def process_chunk(chunk, chunk_id):
    """Processes a dataset chunk, applying AI detection."""
    print(f"🔍 Processing Chunk {chunk_id} with {len(chunk)} rows")
    detectgpt_scores, roberta_ai_probs = [], []
    
    for i, text in enumerate(chunk["text"]):
        detectgpt_score = compute_log_probability_openvino(text)
        roberta_results = detect_ai_openvino_roberta(text)
        
        detectgpt_scores.append(detectgpt_score)
        roberta_ai_probs.append(roberta_results["ai_prob"] if roberta_results else None)
        
        print(f"✅ Row {i+1}/{len(chunk)} - DetectGPT: {detectgpt_score}, RoBERTa: {roberta_ai_probs[-1]}")
    
    chunk["detectgpt_score"], chunk["roberta_ai_prob"] = detectgpt_scores, roberta_ai_probs
    return chunk

if __name__ == "__main__":
    start_time = time.time()
    results, chunk_id = [], 0
    
    with pd.read_csv("scraped_text/dataset.csv", chunksize=CHUNK_SIZE) as reader:
        for chunk in tqdm(reader, desc="🔍 Processing dataset in chunks"):
            if chunk.empty:
                print(f"⚠️ Skipping empty chunk {chunk_id}")
                continue
            
            processed_chunk = process_chunk(chunk, chunk_id)
            results.append(processed_chunk)
            chunk_id += 1
    
    df_final = pd.concat(results, ignore_index=True)
    df_final.to_csv("scraped_text/ai_text_dataset.csv", index=False)
    
    elapsed_time = time.time() - start_time
    print(f"✅ AI detection dataset saved in {elapsed_time:.2f} seconds")

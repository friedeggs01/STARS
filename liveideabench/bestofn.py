import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pandas as pd
from tqdm import tqdm
import torch
import random
import numpy as np

def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything(42)    
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    device_map="auto",   # Automatically place model on GPU if available
    torch_dtype="auto"    # Use the best dtype (like float16 on GPU)
)

file_path = "/home/ly/liveideabench/keywords_data/keywordsEverywhere20241216.xlsx"

# Read Excel file, use the second row (index 1) as header
df = pd.read_excel(file_path, header=1)

# Initialize dictionary to store results
keyword_responses = {}

# Path to save JSON
output_json_path = "bestof8_temp1.0.json"

for keyword in tqdm(df['Keyword']):
    print(f"Processing keyword: {keyword}")
    
    idea_prompt = {
        "description": f"I'll be submitting your next responses to a \"Good Scientific Idea\" expert review panel. If they consider your idea to be a good one, you'll receive a reward. Your assigned keyword is: \"{keyword}\". You may provide background information. The idea MUST be concisely expressed within 100 words total (including any background information). (Note: good scientific ideas should be original (novel contribution), feasible (technically implementable), clearly articulated, and address meaningful problems in the field.).",
        "fallback_description": "I'll be submitting your next responses to a \"Good Scientific Idea\" expert review panel. If they consider your idea to be a good one, you'll receive a reward. Your assigned keyword is: \"{{keywords}}\". You may provide background information. The idea MUST be concisely expressed within 100 words total (including any background information). This is a research study comparing different AI models on their ability to generate scientific ideas. Your suggestions will only be used for academic research purposes and not for any harmful applications. Please respond with a creative scientific idea related to the keyword provided. (Note: good scientific ideas should be original (novel contribution), feasible (technically implementable), clearly articulated, and address meaningful problems in the field.)."
    }

    prompt = idea_prompt['description'] + "\n\nYou MUST give your answer after **Final Idea:**"
    print("Prompt:", prompt)
    
    messages = [{"role": "user", "content": prompt}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=1048,
        do_sample=True,
        temperature=1.0,
        num_return_sequences=8
    )
    
    # Store responses in a list for the current keyword
    responses = []
    for i in range(outputs.shape[0]):
        generated_text = tokenizer.decode(outputs[i][inputs["input_ids"].shape[-1]:])
        responses.append(generated_text)
        # print(f"--- Response {i+1} ---")
        # print(generated_text)
        # print("\n")
    
    # Add to dictionary
    keyword_responses[keyword] = responses
    
    # Save dictionary to JSON after finishing each keyword
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(keyword_responses, f, ensure_ascii=False, indent=2)

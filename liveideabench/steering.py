import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pandas as pd
from tqdm import tqdm
import torch
import random
import numpy as np
import time

def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def ensure_dir(path):
    """Tạo folder nếu chưa có."""
    if not os.path.exists(path):
        os.makedirs(path)   
         
def svd_one_step(H, C=0.1, eps=0):
    H = H.T # now H is d x N
    d, N = H.shape

    Q, Sigma, RT =torch.linalg.svd(H, full_matrices=True) # Sigma here is a 1-D tensor

    # print(f"SVD time: {t2 - t1:.6f} seconds")
    alpha = C * (Sigma[0]**2)
    alpha_sqrt = torch.sqrt(torch.tensor(alpha, device=H.device))
    r = torch.sum(Sigma > eps).item() # rank of H
    Sigma_r = Sigma[:r]
    #choose the last d-r columns of Q
    Q2 = Q[:, r:d] # d x (d-r)
    # randomly select N columns of Q_2
    indices = torch.randperm(d-r)[:N]
    indices, _ = torch.sort(indices)
    V0 = alpha_sqrt * Q2[:, indices] # d x N
    _D = Sigma_r**2 / (Sigma_r**2 + alpha)
    D1 = torch.sum(_D) * 2
    D2 = torch.sum(_D**2) * 4
    eta_opt = D1 / D2
    t_Sigma = 1 / torch.sqrt(alpha + eta_opt**2 * Sigma)
    V1 = alpha_sqrt * (V0 + eta_opt*H) @ RT.T @ torch.diag(t_Sigma) @ RT # d x N
    return V1

def apply_steering_hook_pre(model, tokenizer, recalc_steer_after_n_tokens: int = 1, steer_at_layer: int = 19):
    input_ids = None
    token_cnt = 0
    v_steering = None

    def steering_hook(module, input):
        nonlocal token_cnt, v_steering, input_ids, tokenizer
        token_cnt += 1
        if isinstance(input, tuple):
            input_tensor = input[0]
            other_inputs = input[1:]
        else:
            input_tensor = input
            other_inputs = ()

        if (token_cnt - 2) % recalc_steer_after_n_tokens == 0 or token_cnt == 2:
            h_last_np = input_tensor[:, -1, :].detach().to(torch.float32)
            v_steering = svd_one_step(h_last_np)
            v_steering = torch.tensor(v_steering, dtype=input_tensor.dtype, device=input_tensor.device)
      
        if v_steering is not None:
            input_tensor[:, -1,  :] += v_steering.T
        if isinstance(input, tuple):
            return (input_tensor, *other_inputs)
        else:
            return input_tensor

    steering_handle = model.model.layers[steer_at_layer].self_attn.o_proj.register_forward_pre_hook(steering_hook)
    
    return steering_handle

seed_everything(42)    
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    device_map="auto",   # Automatically place model on GPU if available
    torch_dtype="auto"    # Use the best dtype (like float16 on GPU)
)

file_path = "liveideabench/keywords_data/keywordsEverywhere20241216.xlsx"

# Read Excel file, use the second row (index 1) as header
df = pd.read_excel(file_path, header=1)

# Initialize dictionary to store results
keyword_responses = {}
# steer_at_layer = 19
steer_at_layer = int(os.environ.get("STEER_LAYER"))
num_return_sequences = int(os.environ.get("NUM_RETURN_SEQUENCES"))
temp = float(os.environ.get("TEMP"))
# Path to save JSON
output_dir = "results/steering_layers"
ensure_dir(output_dir)
output_json_path = os.path.join(output_dir, f"steer{num_return_sequences}_temp{temp}_layer{steer_at_layer}.json")

print(f"Steering sequence {num_return_sequences}, temp {temp}, saving to {output_json_path}")
for keyword in tqdm(df['Keyword'][:118]):
    # print(f"Processing keyword: {keyword}")
    
    idea_prompt = {
        "description": f"I'll be submitting your next responses to a \"Good Scientific Idea\" expert review panel. If they consider your idea to be a good one, you'll receive a reward. Your assigned keyword is: \"{keyword}\". You may provide background information. The idea MUST be concisely expressed within 100 words total (including any background information). (Note: good scientific ideas should be original (novel contribution), feasible (technically implementable), clearly articulated, and address meaningful problems in the field.).",
        "fallback_description": "I'll be submitting your next responses to a \"Good Scientific Idea\" expert review panel. If they consider your idea to be a good one, you'll receive a reward. Your assigned keyword is: \"{{keywords}}\". You may provide background information. The idea MUST be concisely expressed within 100 words total (including any background information). This is a research study comparing different AI models on their ability to generate scientific ideas. Your suggestions will only be used for academic research purposes and not for any harmful applications. Please respond with a creative scientific idea related to the keyword provided. (Note: good scientific ideas should be original (novel contribution), feasible (technically implementable), clearly articulated, and address meaningful problems in the field.)."
    }

    prompt = idea_prompt['description'] + "\n\nYou MUST give your answer after **Final Idea:**"
    
    messages = [{"role": "user", "content": prompt}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    handle = apply_steering_hook_pre(model, tokenizer, steer_at_layer=steer_at_layer)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1048,
        do_sample=True,
        temperature=temp,
        num_return_sequences=num_return_sequences
    )
    handle.remove()
    
    # Store responses in a list for the current keyword
    responses = []
    for i in range(outputs.shape[0]):
        generated_text = tokenizer.decode(outputs[i][inputs["input_ids"].shape[-1]:])
        responses.append(generated_text)
    
    # Add to dictionary
    keyword_responses[keyword] = responses
    
    # Save dictionary to JSON after finishing each keyword
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(keyword_responses, f, ensure_ascii=False, indent=2)

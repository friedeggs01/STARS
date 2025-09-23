import os
import transformers
import torch
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import time
from collections import defaultdict  
import json
from multiprocessing import Pool
from functools import partial
import multiprocessing as mp
import random 

from transformers import LlamaForCausalLM, CodeLlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
access_token=os.getenv("HUGGINGFACE_TOKEN")

from data_utils import read_jsonl, write_jsonl, add_lineno

def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def svd_one_step(H, alpha=1, eps=0, C=0.1):
    H = H.T # now H is d x N
    d, N = H.shape
    t1 = time.time()
    Q, Sigma, RT =torch.linalg.svd(H, full_matrices=True) # Sigma here is a 1-D tensor
    t2 = time.time()
    # print(f"SVD time: {t2 - t1:.6f} seconds")
    alpha = C * Sigma[0]**2
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

def apply_steering_hook(model, tokenizer, num_heads = 16, split_id: int = 198, recalc_steer_after_n_tokens: int = 200, steer_at_layer: int = 19):
    input_ids = None
    token_cnt = 0
    v_steering = None
    
    # def get_input_ids_hook(module, input, output):
    #     nonlocal input_ids
    #     # input_ids.shape = [batch size, token length so far]
    #     input_ids = torch.cat((input_ids, input[0].detach().clone()), dim=1) if input_ids is not None else input[0].detach().clone()
    #     # print("Appended tokens:", input_ids[0].tolist())

    def steering_hook(module, input, output):
        nonlocal token_cnt, v_steering, input_ids, tokenizer
        token_cnt += 1
        if input_ids is None or token_cnt < 0: # token_cnt = -1 to skip the first generated token
            return output 

        if isinstance(output, tuple):
            output_tensor = output[0]
            other_outputs = output[1:]
        else:
            output_tensor = output
            other_outputs = ()
         
           
        if (token_cnt - 2) % recalc_steer_after_n_tokens == 0 or token_cnt == 2:
            print(f"Recalculating steering vector at token {token_cnt}")
            h_last_np = output_tensor[:, -1, :].detach().to(torch.float32).cpu().numpy()
            # v_steering, obj_history, grad_norm_history, total_time, step_size_history = profiled_riemannian_descent(h_last_np)
            v_steering = svd_one_step(h_last_np)
            v_steering = torch.tensor(v_steering, dtype=output_tensor.dtype, device=output_tensor.device)
        # breakpoint()    
        if v_steering is not None:
            output_tensor[:, -1,  :] += v_steering.T
        # breakpoint()
        output_tensor = output_tensor.view(output_tensor.shape[0], output_tensor.shape[1], -1)
        if isinstance(output, tuple):
            return (output_tensor, *other_outputs)
        else:
            return output_tensor

    # embed_handle = model.model.model.embed_tokens.register_forward_hook(get_input_ids_hook)
    steering_handle = model.model.model.layers[steer_at_layer].self_attn.o_proj.register_forward_hook(steering_hook)
    # breakpoint()
    
    return steering_handle

def apply_steering_hook_pre(model, tokenizer, num_heads = 16, split_id: int = 198, recalc_steer_after_n_tokens: int = 200, steer_at_layer: int = 19):
    input_ids = None
    token_cnt = 0
    v_steering = None
    
    # def get_input_ids_hook(module, input, output):
    #     nonlocal input_ids
    #     # input_ids.shape = [batch size, token length so far]
    #     input_ids = torch.cat((input_ids, input[0].detach().clone()), dim=1) if input_ids is not None else input[0].detach().clone()
    #     # print("Appended tokens:", input_ids[0].tolist())

    def steering_hook(module, input):
        nonlocal token_cnt, v_steering, input_ids, tokenizer
        token_cnt += 1
        # print(f"Steering hook called at token {token_cnt}")
        # breakpoint()
        # if input_ids is None or token_cnt < 0: # token_cnt = -1 to skip the first generated token
        #     print("Skipping steering hook")
        #     return input 

        if isinstance(input, tuple):
            input_tensor = input[0]
            other_inputs = input[1:]
        else:
            input_tensor = input
            other_inputs = ()
        # breakpoint()
        # print("Recalc check:", (token_cnt - 2) % recalc_steer_after_n_tokens == 0, token_cnt == 2)
        if (token_cnt - 2) % recalc_steer_after_n_tokens == 0 or token_cnt == 2:
            # print(f"Recalculating steering vector at token {token_cnt}")
            # h_last_np = input_tensor[:, -1, :].detach().to(torch.float32).cpu().numpy()
            h_last_np = input_tensor[:, -1, :].detach().to(torch.float32)
            # v_steering, obj_history, grad_norm_history, total_time, step_size_history = profiled_riemannian_descent(h_last_np)
            v_steering = svd_one_step(h_last_np)
            v_steering = torch.tensor(v_steering, dtype=input_tensor.dtype, device=input_tensor.device)
        # breakpoint()    
        if v_steering is not None:
            input_tensor[:, -1,  :] += v_steering.T
        # breakpoint()
        input_tensor = input_tensor.view(input_tensor.shape[0], input_tensor.shape[1], -1)
        if isinstance(input, tuple):
            return (input_tensor, *other_inputs)
        else:
            return input_tensor

    steering_handle = model.model.model.layers[steer_at_layer].self_attn.o_proj.register_forward_pre_hook(steering_hook)
    # breakpoint()
    
    return steering_handle

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='leetcode')
    parser.add_argument("--model", type=str, default='codellama/CodeLlama-7b-Instruct-hf')
    parser.add_argument("--num_tests", type=int, default=10, help='number of tests generated per program')
    parser.add_argument("--temperature", type=float, default=1e-5)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--recalc_steer_after_n_tokens", type=int, default=1, help='recalculate steering vector after generating these many tokens')
    parser.add_argument("--steer_at_layer", type=int, default=23, help='which transformer layer to apply steering')
    return parser.parse_args()

model_list=['codellama/CodeLlama-7b-Instruct-hf','codellama/CodeLlama-13b-Instruct-hf','codellama/CodeLlama-34b-Instruct-hf',
            'meta-llama/Meta-Llama-3-8B-Instruct',
            'bigcode/starcoder2-15b-instruct-v0.1',
            'google/gemma-1.1-2b-it', 'google/gemma-1.1-7b-it'
            'google/codegemma-7b-it',
            'deepseek-ai/deepseek-coder-1.3b-instruct', 'deepseek-ai/deepseek-coder-6.7b-instruct',
            'deepseek-ai/deepseek-coder-33b-instruct',
            'mistralai/Mistral-7B-Instruct-v0.2', 'mistralai/Mistral-7B-Instruct-v0.3'
            'Qwen/CodeQwen1.5-7B-Chat', 'Qwen/Qwen3-1.7B'
            ]

#models do not support system message
models_nosys=['google/gemma-1.1-2b-it','google/gemma-1.1-7b-it',
            'bigcode/starcoder2-15b-instruct-v0.1',
            'mistralai/Mistral-7B-Instruct-v0.3']

def testgeneration_steering(args,prompt,system_message=''):
    """generate test cases with multi-round conversation, each time generate one test case"""
    template_append="Generate another test method for the function under test. Your answer must be different from previously-generated test cases, and should cover different statements and branches."
    generated_tests=[]

    if args.model in models_nosys: #models don't support system message
        messages=[{"role": "user", "content": system_message+prompt}]
    else:
        messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
        
    # print("Messages:", messages)
    handle = apply_steering_hook_pre(generator, generator.tokenizer, recalc_steer_after_n_tokens=args.recalc_steer_after_n_tokens, steer_at_layer=args.steer_at_layer)
    prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    generated=generator(prompt, 
                        max_new_tokens=args.max_tokens, 
                        temperature=args.temperature, 
                        return_full_text=False,
                        num_return_sequences=args.num_tests,
                        do_sample=True)
    generated_tests = [generated[i]['generated_text'] for i in range(len(generated))]
    handle.remove()

    return generated_tests


if __name__=='__main__':
    seed_everything(42)
    mp.set_start_method("spawn", force=True)
    args=parse_args()
    model_abbrv=args.model.split('/')[-1]
    print('Model:', model_abbrv)
    output_dir = Path('predictions_svd_6-qwen')
    
    dataset=read_jsonl('data/leetcode-py.jsonl')

    prompt_template=open('prompt/template_base.txt').read()
    system_template=open('prompt/system.txt').read()
    system_message=system_template.format(lang='python')

    model = AutoModelForCausalLM.from_pretrained(args.model, token=access_token, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=access_token, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
    generator = pipeline("text-generation",model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map='auto', token=access_token)

    data_size=len(dataset)
    # testing_results=[]
    print('number of samples:',len(dataset))
    
    start_time = time.time()
    
    testing_results=[]
    for i in tqdm(range(data_size)):
# for args.steer_at_layer in range(29): #try different layers
    # testing_results=[]
    # for i in tqdm(range(22)):
        data=dataset[i]
        func_name=data['func_name']
        desc=data['description']
        code=data['python_solution']
        difficulty=data['difficulty']
        code_withlineno=add_lineno(code)
        target_lines=data['target_lines']

        prompt=prompt_template.format(lang='python', program=code, description=desc, func_name=func_name)
        generated_tests=testgeneration_steering(args,prompt,system_message)

        testing_data={'task_num':data['task_num'],'task_title':data['task_title'],'func_name':func_name,'difficulty':difficulty,'code':code,'tests':generated_tests}
        testing_results.append(testing_data)
        print('<<<<----------------------------------------->>>>')
        write_jsonl(testing_results,output_dir / f'totalcov_steer_{model_abbrv}_n{args.num_tests}_temp{args.temperature}_anchor{args.recalc_steer_after_n_tokens}_layer{args.steer_at_layer}_C0.1_preoproj.jsonl')
        
    end_time = time.time()
    total_time = end_time - start_time
        # Save log file
    log_data = {
        "result_file": f'totalcov_steer_{model_abbrv}_n{args.num_tests}_temp{args.temperature}_anchor{args.recalc_steer_after_n_tokens}_layer{args.steer_at_layer}_C0.1_preoproj.jsonl',
        "number_of_samples": data_size,
        "args": vars(args),
        "total_time_seconds": total_time
    }
    log_file = output_dir / f'runlog_totalcov_steer_{model_abbrv}_n{args.num_tests}_temp{args.temperature}_anchor{args.recalc_steer_after_n_tokens}_layer{args.steer_at_layer}_C0.1_preoproj.json'
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    write_jsonl(testing_results,output_dir / f'totalcov_{model_abbrv}.jsonl')

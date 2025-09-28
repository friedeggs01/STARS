import os
import torch
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import time
import json
from transformers import LlamaForCausalLM, CodeLlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
access_token=os.getenv("HUGGINGFACE_TOKEN")
import random
import numpy as np
from data_utils import read_jsonl, write_jsonl, add_lineno


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='leetcode')
    parser.add_argument("--model", type=str, default='codellama/CodeLlama-7b-Instruct-hf')
    parser.add_argument("--num_tests", type=int, default=10, help='number of tests generated per program')
    parser.add_argument("--temperature", type=float, default=1e-5)
    parser.add_argument("--max_tokens", type=int, default=256)
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
models_nosys=['google/gemma-1.1-2b-it', 'google/gemma-1.1-7b-it',
            'bigcode/starcoder2-15b-instruct-v0.1',
            'mistralai/Mistral-7B-Instruct-v0.3']

def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def testgeneration_multiround(args,prompt,system_message=''):
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

    for i in range(args.num_tests):
        prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        generated=generator(prompt, 
                            max_new_tokens=args.max_tokens, 
                            temperature=args.temperature, 
                            return_full_text=False)
        
        generated_test=generated[0]['generated_text']
        print(generated_test)

        messages.append({"role": "assistant", "content": generated_test})
        messages.append({"role": "user", "content": template_append})
        

        generated_tests.append(generated_test)
    return generated_tests

def testgeneration_bestofn(args,prompt,system_message=''):
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

    prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    generated=generator(prompt, 
                        max_new_tokens=args.max_tokens, 
                        temperature=args.temperature, 
                        return_full_text=False,
                        num_return_sequences=args.num_tests,
                        do_sample=True)
    generated_tests = [generated[i]['generated_text'] for i in range(len(generated))]
    # print(generated_tests)

    return generated_tests


if __name__=='__main__':
    args=parse_args()
    seed_everything(42)
    model_abbrv=args.model.split('/')[-1]
    print('Model:', model_abbrv)
    output_dir = Path('predictions')
    
    dataset=read_jsonl('data/leetcode-py.jsonl')

    prompt_template=open('prompt/template_base.txt').read()
    system_template=open('prompt/system.txt').read()
    system_message=system_template.format(lang='python')

    model = AutoModelForCausalLM.from_pretrained(args.model, token=access_token, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=access_token, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
    generator = pipeline("text-generation",model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map='auto', token=access_token)

    data_size=len(dataset)
    testing_results=[]
    print('number of samples:',len(dataset))
    
    start_time = time.time()
    for i in tqdm(range(data_size)):
        data=dataset[i]
        func_name=data['func_name']
        desc=data['description']
        code=data['python_solution']
        difficulty=data['difficulty']
        code_withlineno=add_lineno(code)
        target_lines=data['target_lines']

        prompt=prompt_template.format(lang='python', program=code, description=desc, func_name=func_name)
        generated_tests=testgeneration_bestofn(args,prompt,system_message)

        testing_data={'task_num':data['task_num'],'task_title':data['task_title'],'func_name':func_name,'difficulty':difficulty,'code':code,'tests':generated_tests}
        testing_results.append(testing_data)
        print('<<<<----------------------------------------->>>>')
        write_jsonl(testing_results,output_dir / f'totalcov_bestofn_{model_abbrv}_n{args.num_tests}_temp{args.temperature}_42.jsonl')

    end_time = time.time()
    total_time = end_time - start_time
        # Save log file
    log_data = {
        "result_file": f'totalcov_bestofn_{model_abbrv}_n{args.num_tests}_temp{args.temperature}_42.jsonl',
        "number_of_samples": data_size,
        "args": vars(args),
        "total_time_seconds": total_time
    }
    log_file = output_dir / f'runlog_totalcov_steer_{model_abbrv}_n{args.num_tests}_temp{args.temperature}_42.json'
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

    write_jsonl(testing_results,output_dir / f'totalcov_{model_abbrv}.jsonl')

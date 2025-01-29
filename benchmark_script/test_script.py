import json
from datetime import datetime
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM


def read_json_input(input_file):
    with open(input_file, 'r') as file:
        data = json.load(file)
    return [item['prompt'] for item in data]


def generate_prompts(model, tokenizer, prompts, device='cpu', max_length=100, temperature=1.0, top_p=0.01,
                     repetition_penalty=1.0):
    model.to(device)
    model.eval()
    generated_texts = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=max_length, temperature=temperature, top_p=top_p,
                                     repetition_penalty=repetition_penalty)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(generated_text)
    return generated_texts


def run_inference(config):
    model_name_or_path = config['model_name_or_path']
    input_file = config['input_file']
    output_file = config['output_file']
    results_file = config.get('results_file')
    device = config['device']
    batch_size = config['batch_size']
    max_seq_len = config['max_seq_len']
    temperature = config['temperature']
    top_p = config['top_p']
    repetition_penalty = config['repetition_penalty']


    prompts = read_json_input(input_file)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    all_generated_texts = []
    total_time = 0  # Total time for all requests
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        start_time = time.time()
        generated_texts = generate_prompts(model, tokenizer, batch_prompts, device=device, max_length=max_seq_len,
                                           temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)
        end_time = time.time()
        batch_time = end_time - start_time
        total_time += batch_time
        all_generated_texts.extend(generated_texts)

    with open(output_file, 'w') as f:
        json.dump(all_generated_texts, f, indent=4)

    request_cost = total_time / len(prompts)  # Average time per request
    one_batch_cost = total_time / (len(prompts) / batch_size)  # Time per batch
    average_cost = total_time / len(all_generated_texts)  # Average time per generated text

    with open(results_file, 'w') as f:
        f.write(f"Request cost: {request_cost} sec\n")
        f.write(f"One batch cost: {one_batch_cost} sec\n")
        f.write(f"Average cost: {average_cost} sec\n")

    print(f"Output saved to {output_file}")


def load_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config


if __name__ == "__main__":
    config = load_config()
    run_inference(config)
    print("Inference completed:", datetime.now())
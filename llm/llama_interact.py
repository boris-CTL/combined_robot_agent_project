import numpy as np
import torch
from fastchat.model import get_conversation_template
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLM:
    def __init__(self, model_name="gpt2",
                 tokenizer=None,
                 model=None,
                 encoder_decoder=False,
                 use_fastchat_model=False,
                 device="cuda",
                 device_map="auto"):
        if not tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer
        self.encoder_decoder = encoder_decoder
        self.device = device
        if not model:
            if encoder_decoder:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                                   device_map=device_map)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                                  device_map=device_map)
        else:
            self.model = model
            self.model = self.model.to(self.device)

        self.model_name = model_name
        self.use_fastchat_model = use_fastchat_model
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'right'

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def input_encode(self, input_sent: str):
        if self.use_fastchat_model:
            conv = get_conversation_template(self.model_path)
            conv.append_message(conv.roles[0], input_sent)
            conv.append_message(conv.roles[1], None)
            input_sent = conv.get_prompt()
        tensor_input = self.tokenizer.encode(input_sent, return_tensors='pt').to(self.device).to(self.model.dtype)
        return tensor_input

    def __call__(self, input_sent: str,
                 do_sample=False,
                 top_k=50,
                 top_p=0.95,
                 typical_p=1.0,
                 no_repeat_ngram_size=0,
                 temperature=1.0,
                 repetition_penalty=1.0,
                 guidance_scale=1,
                 max_new_tokens=512):

        tokenized = self.tokenizer(input_sent, padding=True, return_tensors='pt')
        input_ids = tokenized.input_ids.to(self.device)

        output_ids = self.model.generate(
            input_ids,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            typical_p=typical_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            guidance_scale=guidance_scale,
            max_new_tokens=max_new_tokens
        )

        actual_seq_lengths = tokenized.attention_mask.sum(dim=1)
        output_ids = [output_id[seq_length:] for output_id, seq_length in zip(output_ids, actual_seq_lengths)]

        predictions = []
        for prediction in self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        ):
            prediction = prediction.strip()
            predictions.append(prediction)
        return predictions
  
def init(cfg):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    # 'meta-llama/Llama-2-7b-chat-hf'
    # 'codellama/CodeLlama-7b-hf'
    print(f"Using model: {cfg['model_name']}")
    llm = LLM(model_name=cfg['model_name'], device=device)
    return llm


def produce_program(logger, task_desc, program_num, init_state_str, llm):
    # Produce program using task description, program will first be parsed then return
    with open('llm/system_prompt.txt', 'r') as file:
        system_prompt_contents = file.read()
    
    with open('llm/state_prompt.txt', 'r') as file:
        state_prompt_contents = file.read()

    system_prompt = f"""{system_prompt_contents}"""

    user_prompt = f"""Given the task description for Karel programming:
{task_desc}


The state explanation for karel is as follows:
{state_prompt_contents}

Given the initial state as follows:
{init_state_str}

Please generate {program_num} unique Karel programs that could potentially solve the task. Ensure that each program is formatted on a single line and adheres to the correct Karel programming grammar.

Program 1:"""

    # messages = []
    # messages.append({"role": "system", "content": system_prompt})
    # messages.append({"role": "user", "content": user_prompt})
    # messages = llm.tokenizer.apply_chat_tokenization(messages
    messages = f"""{system_prompt}\n{user_prompt}"""
    response = llm(messages, do_sample=True, top_k=10, top_p=0.95, temperature=0.1, max_new_tokens=2048)

    logger.debug(f"System Prompt: {system_prompt}")
    logger.debug(f"User Prompt: {user_prompt}")
    logger.debug(f"LLM Response: {response}")

    response = response[0] # Only take the first response
    response = response.replace("\n", "") # Remove newlines

    start_pos = response.find("DEF")
    end_pos = response.find("m)") + 2

    programs = []

    while start_pos != -1 and end_pos != -1:
        program = response[start_pos: end_pos]
        programs.append(program)

        response = response[end_pos:]
        start_pos = response.find("DEF")
        end_pos = response.find("m)") + 2
    
    
    logger.debug(f"There are {len(programs)} generated programs :\n{programs}")
    return programs

def produce_feedback_programs(logger, task_desc, program_num, init_state_str, trajectories_str, previous_programs_str, llm):
    with open('llm/system_prompt.txt', 'r') as file:
        system_prompt_contents = file.read()
    
    system_prompt = f"""{system_prompt_contents}"""
    
    with open('llm/state_prompt.txt', 'r') as file:
        state_prompt_contents = file.read()

    # Given previous trajectories, produce feedback programs for better trajectories
    user_prompt = f"""Given the task description for Karel programming:
{task_desc}

The state explanation for karel is as follows:
{state_prompt_contents}

Here are some previous trajectories that were generated by an agent:
{trajectories_str}

Given these trajectories, please reflect on the agent's performance. 

Previous programs:
{previous_programs_str}

Given the initial state as follows:
{init_state_str}

Please generate {program_num} unique Karel programs that could potentially solve the task. Ensure that each program is formatted on a single line and adheres to the correct Karel programming grammar.\n\n

Program 1:"""

    # Previous programs

    # user_prompt += "The response should be format as follows: \n"
    # user_prompt += "REFLACTION: <your reflection on the agent's performance>\n"
    # user_prompt += "PROGRAMS:\n1. <your program>\n2. <your program>\n3. <your program>\n...\n"

    # messages = []
    # messages.append({"role": "system", "content": system_prompt})
    # messages.append({"role": "user", "content": user_prompt})
    # messages = llm.tokenizer.apply_chat_tokenization(messages)
    messages = f"""{system_prompt}\n{user_prompt}"""
    response = llm(messages, do_sample=True, top_k=10, top_p=0.95, temperature=0.1, max_new_tokens=2048)

    logger.debug(f"System Prompt: {system_prompt}")
    logger.debug(f"User Prompt: {user_prompt}")
    logger.debug(f"LLM Response: {response}")

    response = response[0] # Only take the first response
    response = response.replace("\n", "")

    start_pos = response.find("DEF")
    end_pos = response.find("m)") + 2

    programs = []

    while start_pos != -1 and end_pos != -1:
        program = response[start_pos: end_pos]
        programs.append(program)

        response = response[end_pos:]
        start_pos = response.find("DEF")
        end_pos = response.find("m)") + 2
    
    
    logger.debug(f"There are {len(programs)} generated programs :\n{programs}")
    return programs

if __name__ == "__main__":
    import time
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    # 'meta-llama/Llama-2-7b-chat-hf'
    # 'codellama/CodeLlama-7b-hf'
    llm = LLM(model_name='meta-llama/Llama-2-7b-chat-hf', device=device)

    with open('env_func.py', 'r') as file:
        prompt = file.read()

    print("Starting timer...")
    start_time = time.time()
    print(llm(prompt, do_sample=True, top_k=10, top_p=0.95, temperature=0.1, max_new_tokens=2048))
    print(time.time() - start_time)
# 20240314, use gpt-3.5-turbo to unroll step-by-step, llf-metaworld-v2-reach-v2 env.
# execute_Karel_program_string_BorVersion.py's import
import time
import argparse
import os
import logging
import importlib.util
import sys
import copy

from tensorboardX import SummaryWriter
import wandb
from wandb.integration.sb3 import WandbCallback
import torch
import gymnasium as gym
from stable_baselines3 import A2C, PPO, DQN


from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import random
import pickle
import shutil
import imageio
from PIL import Image
from pygifsicle import optimize
import tqdm as tqml



from torch.utils.data.dataset import Dataset, random_split
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from cfg import config


# import gym
import gymnasium
from io import BytesIO



# llama2chat's import, class, functions
from fastchat.model import get_conversation_template
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForCausalLM

# 20240216
import re


# from gym.wrappers.monitoring.video_recorder import VideoRecorder
import llfbench as gym_
import matplotlib.pyplot as plt
import random
import math
from gymnasium.envs.registration import register



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
  

import openai
import traceback

def openai_chat_response(messages, temperature = 1.0, gpt="gpt-3.5-turbo"): #gpt="gpt-4"):
    print(f'Prompt:\n{messages}')
    # input('Continue?\n')
    
    try:
        openai.api_key = "your-api-key-here"
        response = openai.ChatCompletion.create(
        model=gpt,
        messages=messages,
        temperature=temperature
        )
        return response.choices[0].message.content
    except:
        traceback.print_exc()
        time.sleep(100)


# 20240314, done modified
def extract_llm_response(text):
    # Use regular expression to extract the action information
    result = re.findall(r'\[([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+)\]', text)
    # print(result)
    
    # Convert the extracted result to a list of floats
    extracted_result = [float(x) for x in result[0]]
    
    return extracted_result

def init(cfg):
    # Set up OpenAI api key
    openai.api_key = cfg['api_key']
    if cfg['openai_org'] is not None:
        openai.organization = cfg['openai_org'] 

# should be done modified.
def produce_initial_response(logger, initial_state_observation_string, b_instruction, gpt = "gpt-3.5-turbo", temperature = 0.3):
    with open('llm/system_prompt_llfMetaWorldReachV2.txt', 'r') as file:
        system_prompt_contents = file.read()

    system_prompt = f"""{system_prompt_contents}"""

    user_prompt = f"""
Given the reach task instruction as your system prompt, the initial observation (state) is :

{initial_state_observation_string}

{b_instruction}
"""

    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    response = openai_chat_response(messages=messages, temperature=temperature, gpt=gpt)

    logger.debug(f"System Prompt: {system_prompt}")
    logger.debug(f"User Prompt: {user_prompt}")
    logger.debug(f"LLM Response: {response}")

    # llm may output nonsense rubbish. In this case, calling `extract_llm_response` will fail.
    try:
        # extract llm response
        extracted_result = extract_llm_response(response)
        logger.debug(f"The extracted result from llm's response is :\n{extracted_result}")
        return extracted_result
    except IndexError:
        print("llm outputted rubbish in initial response.")
        return -27
    



# should be done modified.
def produce_feedback_response(logger, current_state_observation_string, b_instruction, r_hp_hn_feedback, gpt = "gpt-3.5-turbo", temperature = 0.3):
    with open('llm/system_prompt_llfMetaWorldReachV2.txt', 'r') as file:
        system_prompt_contents = file.read()

    system_prompt = f"""{system_prompt_contents}"""

    user_prompt = f"""
Given the reach task instruction as your system prompt, the initial observation (state) is :

{current_state_observation_string}

{r_hp_hn_feedback} This reward is the result of your past actions.\n

{b_instruction}
"""

    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    response = openai_chat_response(messages=messages, temperature=temperature, gpt=gpt)

    logger.debug(f"System Prompt: {system_prompt}")
    logger.debug(f"User Prompt: {user_prompt}")
    logger.debug(f"LLM Response: {response}")
    try:
        # extract llm response
        extracted_result = extract_llm_response(response)
        logger.debug(f"The extracted result from llm's response is :\n{extracted_result}")
        return extracted_result
    except IndexError:
        print("llm outputted rubbish in feedback responses.")
        return -27
    
def parse_config(configfile):
    spec = importlib.util.spec_from_file_location('cfg', configfile)
    conf_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf_mod)
    config = conf_mod.config
    print(config)
    return config

def parse_traj_list(global_traj_list_):
    all_state_list = []
    all_action_list = []
    all_reward_list = []
    for i in range(len(global_traj_list_)):
        if (i % 3 == 0):
            # it's a state
            all_state_list.append(global_traj_list_[i])
        elif (i % 3 == 1):
            # it's an action
            all_action_list.append(global_traj_list_[i])
        elif (i % 3 == 2):
            # it's a reward
            all_reward_list.append(global_traj_list_[i])
    return all_state_list, all_action_list, all_reward_list



if __name__ == "__main__":

    t_init = time.time()

    # Parse the configfile first
    config = parse_config('cfg.py')


    # init metaworld-reach-v2 env, setting seed, task_idx
    env_name = "reach-v2"
    SEED_NUMBER = 666
    task_idx = 6

    register(
        id=f"llf-metaworld-{env_name}",
        entry_point='llfbench.envs.metaworld:make_env',
        kwargs=dict(env_name=env_name, feedback_type='a', instruction_type='b', SEED=SEED_NUMBER, task_idx=task_idx)
    )

    env = gym_.make(f"llf-metaworld-{env_name}")

    done = False
    cumulative_reward = 0.0

    # First observation is acquired by resetting the environment
    observation, info = env.reset()

    # create the output directory
    outdir = f"./outdir_{env_name}"
    # Create output directory if it does not already exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Set up logger
    log_file = os.path.join(outdir, f"logfile_seed{SEED_NUMBER}_idx{task_idx}.log")
    log_handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode='w')]
    
    logging.basicConfig(handlers=log_handlers, format=config['logging']['fmt'], level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    print(config['logging'])
    logger.setLevel(logging.getLevelName(config['logging']['level']))
    # logger.disabled = (not config['verbose'])


    # set up OpenAI key
    config['api_key'] = "your api key here"
    config['openai_org'] = "org-VFSaZQgJUGtyglfUzEzesHAC"
    
    
    # llm model initialisation in __main__ :
    init(config)

    # this b_intruction is added by myself. boris.
    # by modifying the b_instruction provided in LLF-Bench's other tasks.
    b_instruction = (
        "Output a good 4-dim vector in the form of [x, y, z, gripper_state].",
        "Render a proper 4-dim vector in the style of [x, y, z, gripper_state].",
        "Supply a good 4-dim vector in the form of [x, y, z, gripper_state].",
        "Indicate a good 4-dim vector using [x, y, z, gripper_state] as the format.",
        "Come up with a good 4-dim vector in the form of [x, y, z, gripper_state].",
        "Present a correct 4-dim vector in the form of [x, y, z, gripper_state].",
    )
    selected_instruction = random.choice(b_instruction)

    
    global_traj_list = []
    global_traj_list.append(env.current_observation)
    
    action = produce_initial_response(logger, str(observation['observation']), str(selected_instruction), gpt = "gpt-3.5-turbo", temperature = 0.3)

    if (action != -27):
        env.render()
        observation, reward, terminated, truncated, info = env.step(action)
        global_traj_list.extend(info['traj_list'])
    elif (action == -27):
        # llm outputted rubbish. choose initial action randomly.
        env.render()
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        global_traj_list.extend(info['traj_list'])

    cumulative_reward += reward

    # terminated and truncated follow the same semantics as in Gymnasium
    done = terminated or truncated

    cnt = 0
    succeeded = False
    
    while ((not done) and (cnt <= 300) and (not succeeded)):
        selected_instruction = random.choice(b_instruction)
        further_action = produce_feedback_response(logger, str(observation['observation']), str(selected_instruction), str(observation['feedback']), gpt = "gpt-3.5-turbo", temperature = 0.3)

        if (further_action != -27):
            img = env.render()
            observation, reward, terminated, truncated, info = env.step(further_action)
            global_traj_list.extend(info['traj_list'])
            
            if (info['success'] == True):
                succeeded = True
        
        
            cumulative_reward += reward
        
            # terminated and truncated follow the same semantics as in Gymnasium
            done = terminated or truncated
    
            cnt += 1
        else:
            # llm outputted rubbish.
            img = env.render()
            cnt += 1



    print("done :", done)
    print("success :", info['success'])
    
    all_s, all_act, all_r = parse_traj_list(global_traj_list)

    # Saving multiple lists to a single .npz file
    np.savez(f'{outdir}/ReachV2_4traj_seed{SEED_NUMBER}_idx{task_idx}.npz', all_traj=global_traj_list, all_states=all_s, all_actions=all_act, all_rewards=all_r)

    # Final time
    t_final = time.time()
    logger.debug('{} Program finished in {} secs.'.format(__name__, t_final - t_init))
    print('{} Program finished in {} secs.'.format(__name__, t_final - t_init))


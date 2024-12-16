# 20240411, use gpt-3.5-turbo to unroll step-by-step, pick-place-v2 env.
import time
import os
import logging
import importlib.util
import sys

from tensorboardX import SummaryWriter
import gymnasium as gym
import numpy as np
import random
from cfg import config
from io import BytesIO
from fastchat.model import get_conversation_template
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForCausalLM

# 20240216
import re


import metaworld
import random
import time
import numpy as np


def extract_llm_response(given_string):
    pattern = r'\[(.*?)\]'  # Regular expression pattern to match text inside square brackets
    matches = re.findall(pattern, given_string)
    if matches:
        # 反向遍歷matches
        for i in range(len(matches) - 1, -1, -1):
            # numbers = re.findall(r'[-+]?\d*\.\d+|\d+', matches[i])  # Extracting numbers from the output
            numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', matches[i])
            response = [float(num) for num in numbers]  # Converting extracted numbers to float
            # print(response)
            if (len(response) == 4):
                break
        if (len(response) == 4):
            return response
        else:
            return []
    else:
        return []

import gymnasium as gym
import metaworld
import random
import time
import importlib
from llfbench.envs.metaworld.gains import P_GAINS
from metaworld.policies.policy import move
from metaworld.policies.action import Action
from metaworld.policies import SawyerDrawerOpenV1Policy, SawyerDrawerOpenV2Policy, SawyerReachV2Policy
import numpy as np

class WrappedSawyerV2Env(gym.Wrapper):
    
    def __init__(self, env, env_name):
        super().__init__(env)
        # load the scripted policy
        if env_name=='peg-insert-side-v2':
            module = importlib.import_module(f"metaworld.policies.sawyer_peg_insertion_side_v2_policy")
            self._policy = getattr(module, f"SawyerPegInsertionSideV2Policy")()
        else:
            module = importlib.import_module(f"metaworld.policies.sawyer_{env_name.replace('-','_')}_policy")
            print(module)
            self._policy = getattr(module, f"Sawyer{env_name.title().replace('-','')}Policy")()
            # print(self._policy)
        self.p_control_time_out = 20 # timeout of the position tracking (for convergnece of P controller)
        self.p_control_threshold = 1e-4 # the threshold for declaring goal reaching (for convergnece of P controller)
        self._current_observation = None
        self.env_name = env_name
        self.consecutive_iterations = 10
        self.dif_threshold = 1e-3
        
    # test function for boris
    def get_attri_boris(self):
        print("self._policy : ", self._policy)
        print("self._current_observation : ", self._current_observation)
        return 27

    @property
    def mw_policy(self):
        return self._policy

    @property
    def current_observation(self):  # external interface
        """ This is a cache of the latest (raw) observation. """
        # print("current_observation() in WrappedSawyerV2Env() is called.")
        # print("self._current_observation is :", self._current_observation)
        return self._current_observation
        
    @property
    def _current_pos(self):
        """ Curret position of the hand. """
        # print("_current_pos() in WrappedSawyerV2Env() is called.")
        # print("The return val of _current_pos() is :", self._policy._parse_obs(self.current_observation)['hand_pos'])
        # return self.mw_policy._parse_obs(self.current_observation)['hand_pos']
        return self.mw_policy._parse_obs(self.current_observation)['hand_pos']


    def check_convergence(self, y_values, epsilon=1e-3, consecutive_iterations=10):
        """
        Check convergence of a list of y-values using a threshold-based method.
        
        Args:
        - y_values: List of y-values
        - epsilon: Threshold value for convergence
        - consecutive_iterations: Number of consecutive iterations for which the difference
                                  must remain below the threshold
        
        Returns:
        - Boolean value indicating whether convergence is achieved
        """
        consecutive_below_threshold = 0
        print("type(y_values) : ", type(y_values))
        prev_y = y_values[0]
        
        for y in y_values[1:]:
            if abs(y - prev_y) < epsilon:
                consecutive_below_threshold += 1
                if consecutive_below_threshold >= consecutive_iterations:
                    return True, y_values.index(y)
            else:
                consecutive_below_threshold = 0
            
            prev_y = y
    
        
        return False, None

    def p_control(self, action):
        """ Compute the desired control based on a position target (action[:3])
        using P controller provided in Metaworld."""
        assert len(action)==4
        # print(type(self._policy))
        # p_gain = P_GAINS[type(self._policy)]
        # print(p_gain)
        p_gain = P_GAINS[type(self.mw_policy)]
        if type(self.mw_policy) in [type(SawyerDrawerOpenV1Policy), type(SawyerDrawerOpenV2Policy)]:
            # This needs special cares. It's implemented differently.
            o_d = self.mw_policy._parse_obs(self.current_observation)
            pos_curr = o_d["hand_pos"]
            pos_drwr = o_d["drwr_pos"]
            # align end effector's Z axis with drawer handle's Z axis
            if np.linalg.norm(pos_curr[:2] - pos_drwr[:2]) > 0.06:
                p_gain = 4.0
            # drop down to touch drawer handle
            elif abs(pos_curr[2] - pos_drwr[2]) > 0.04:
                p_gain = 4.0
            # push toward a point just behind the drawer handle
            # also increase p value to apply more force
            else:
                p_gain= 50.0

        control = Action({"delta_pos": np.arange(3), "grab_effort": 3})
        # check if doing `move` properly
        print("now, self._current_pos is :", self._current_pos)
        print("now, action[:3] is :", action[:3])
        action_should_be = (action[:3] - self._current_pos) * p_gain
        print("manually calculated res of action_should_be is : ", action_should_be)
        control["delta_pos"] = move(self._current_pos, to_xyz=action[:3], p=p_gain)
        print("control[\"delta_pos\"] is :", control["delta_pos"])
        control["grab_effort"] = action[3]
        return control.array


    # here, action is viewed as the desired position + grab_effort
    def _step(self, action):
        # Run P controller until convergence or timeout
        # action is viewed as the desired position + grab_effort
        previous_pos = self._current_pos  # the position of the hand before moving
        
        traj_list_micro = []
        traj_list_macro = []
        traj_list_macro_for_LLM = [action]

        still_be_stepping = True
        step_count = 0
        dif_list = []
        done = False
        # for i in range(self.p_control_time_out):
        while (still_be_stepping):
            control = self.p_control(action)
            observation, reward, terminated, truncated, info = self.env.step(control)
            step_count += 1
            env.render()
            
            traj_list_micro.append(control)
            traj_list_micro.append(reward)
            traj_list_micro.append(observation)
            
            print("the `control` being executed is :", control)
            print("after doing `control`, obs :", observation)
            print("after doing `control`, reward :", reward)

            
            if (terminated or truncated):
                print("after doing `control`, terminated :", terminated)
                print("after doing `control`, truncated :", truncated)
                done = terminated or truncated
                  
            self._current_observation = observation
            desired_pos = action[:3]
            print("np.abs(desired_pos - self._current_pos).max() :", np.abs(desired_pos - self._current_pos).max())
            dif_list.append(np.abs(desired_pos - self._current_pos).max())
            print("dif_list : ", dif_list)

            # check convergence
            if (len(dif_list) >= self.consecutive_iterations + 1):
                print("len(dif_list) is lenghty enough with :", len(dif_list))
                if_converged, y_idx =  \
                    self.check_convergence(dif_list, epsilon=self.dif_threshold, consecutive_iterations=self.consecutive_iterations)
                print("if_converged, y_idx : ", if_converged, y_idx)
                # if_converged, y_idx = \
                #     self.check_convergence(list(dif_list))
                if (if_converged):
                    still_be_stepping = False
            # if np.abs(desired_pos - self._current_pos).max() < self.p_control_threshold:
            #     still_be_stepping = False
                # break
            if (step_count == 1):
                traj_list_macro.append(control)
            if ((not still_be_stepping) or (reward == 10) or (done)):
                # 表示這時候是最後了，不會再進行下一輪step()，故macro要存
                traj_list_macro.append(reward)
                traj_list_macro.append(observation)

                traj_list_macro_for_LLM.append(reward)
                traj_list_macro_for_LLM.append(observation)
                break
            # 假設都已經truncated due to maximum steps reached, 還沒有收斂，那就強制return
            # if (done):
                
            # if (reward == 10):
            #     # 表示這時候雖然可能不是(i == self.p_control_time_out - 1)，但macro還是要存
            #     traj_list_macro.append(reward)
            #     traj_list_macro.append(observation)
                


        info['success'] = bool(info['success'])
        info['traj_list_micro'] = traj_list_micro
        info['traj_list_macro'] = traj_list_macro
        info['traj_list_macro_for_LLM'] = traj_list_macro_for_LLM
        info['step_count'] = step_count
        info['dif_list'] = dif_list
        return observation, reward, terminated, truncated, info
    

    def _reset(self, *, seed=None, options=None):
            self._current_observation, info = self.env.reset(seed=None, options=options)
            print("after reset, the _current_observation is :", self._current_observation)
            # info['success'] = False
            return self._current_observation, info

    def normal_step(self, action):
        return self.env.step(action)



def Make_Wrapped_Sawyer_Env_for_LLM(env_name, SEED_NUMBER, task_idx):
    """ Make the original env and wrap it with the WrappedSawyerV2Env wrapper. """
    mt1 = metaworld.MT1(env_name, seed = SEED_NUMBER)
    env = mt1.train_classes[env_name]()
    task = mt1.train_tasks[task_idx]
    env.set_task(task)  # Set task
    
    env = WrappedSawyerV2Env(env, env_name)
    return env

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

def openai_chat_response(messages, open_ai_api_key, temperature = 1.0, gpt="gpt-3.5-turbo"): #gpt="gpt-4"):
    print(f'Prompt:\n{messages}')
    # input('Continue?\n')
    
    try:
        # openai.api_key = "your-api-key-here"
        openai.api_key = open_ai_api_key
        response = openai.ChatCompletion.create(
        model=gpt,
        messages=messages,
        temperature=temperature
        )
        return response.choices[0].message.content
    except:
        traceback.print_exc()
        time.sleep(100)

def init(cfg):
    # Set up OpenAI api key
    openai.api_key = cfg['api_key']
    if cfg['openai_org'] is not None:
        openai.organization = cfg['openai_org'] 

# should be done modified.
def produce_initial_response(logger, initial_state_info, b_instruction, open_ai_api_key, gpt = "gpt-3.5-turbo", temperature = 0.3):
    global system_prompt_path
    with open(system_prompt_path, 'r') as file:
        system_prompt_contents = file.read()

    system_prompt = f"""{system_prompt_contents}"""

    # pre-process initial_state_info
    hand_pos_init = initial_state_info['hand_pos']
    puck_pos_init = initial_state_info['puck_pos']
    goal_pos_init = initial_state_info['goal_pos']
    
    user_prompt = f"""
Now, given the task instruction, in the initial observation (state) :

(1) `hand_pos` is {hand_pos_init}, representing the x, y and z position of robot's end-effector.
(2) `puck_pos` is {puck_pos_init}, representing the x, y and z position of the puck.
(3) `goal_pos` is {goal_pos_init}, representing the x, y and z position of the goal.

Q1: How do you think of `hand_pos`, `puck_pos` and `goal_pos`? What role does each of them play in this environment?
Q2: Given the ultimate goal of this task, decompose the task goal into several sub-goals. Do you think you have already accomplished any sub-goals? Identify your current situation. In which phase are you currently? Under what conditions will you enter the next phase?
Q3: Based on your answer of Q2, what is the position that you want to move the end-effector to, now?
Q4: {b_instruction}
Note that, you not only need to explain how to obtain the [x, y, z, gripper_state] as your output, you also need to EXPLICITLY give me the output in the form of [x, y, z, gripper_state].
"""

    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    response = openai_chat_response(messages=messages, open_ai_api_key=open_ai_api_key, temperature=temperature, gpt=gpt)

    logger.debug(f"System Prompt: {system_prompt}")
    logger.debug(f"User Prompt: {user_prompt}")
    logger.debug(f"LLM Response: {response}")


    extracted_result = extract_llm_response(response)
    return extracted_result
    
    
# should be done modified.
def produce_feedback_response(logger, current_state_info, b_instruction, reward_feedback, past_traj_str, open_ai_api_key, gpt = "gpt-3.5-turbo", temperature = 0.3):
    global system_prompt_path
    with open(system_prompt_path, 'r') as file:
        system_prompt_contents = file.read()

    system_prompt = f"""{system_prompt_contents}"""

    # pre-process current_state_info
    hand_pos_current = current_state_info['hand_pos']
    puck_pos_current = current_state_info['puck_pos']
    goal_pos_current = current_state_info['goal_pos']

    user_prompt = f"""
Now, given the task instruction, in the current observation (state) :

(1) `hand_pos` is {hand_pos_current}, representing the x, y and z position of robot's end-effector.
(2) `puck_pos` is {puck_pos_current}, representing the x, y and z position of the puck.
(3) `goal_pos` is {goal_pos_current}, representing the x, y and z position of the goal.
(4) Following are the entire past trajectories based on all of your previous actions :
{past_traj_str}

Q1: How do you think of `hand_pos`, `puck_pos` and `goal_pos`? What role does each of them play in this environment?
Q2: Given the ultimate goal of this task, decompose the task goal into several sub-goals. Do you think you have already accomplished any sub-goals? Identify your current situation. In which phase are you currently? Under what conditions will you enter the next phase?
Q3: Reason about the given past trajectories. What do you see in the past trajectories? Did you make any mistakes? If you do think you have made some mistakes, assume you can start over and come up with a method to handle those mistakes. If you think you have done pretty well, keep doing your thing. Integrate this thought into Q4.
Q4: Based on your answer of Q2 and Q3, what is the position that you want to move the end-effector to, now?
Q5: {b_instruction}
Note that, you not only need to explain how to obtain the [x, y, z, gripper_state] as your output, you also need to EXPLICITLY give me the output in the form of [x, y, z, gripper_state].
"""


    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    response = openai_chat_response(messages=messages, open_ai_api_key=open_ai_api_key, temperature=temperature, gpt=gpt)

    logger.debug(f"System Prompt: {system_prompt}")
    logger.debug(f"User Prompt: {user_prompt}")
    logger.debug(f"LLM Response: {response}")
    
        
    extracted_result = extract_llm_response(response)
    return extracted_result

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


def generate_trajectory_string(trajectory_list):
    global env
    policy = env.mw_policy
    # policy._parse_obs(observation)
    result = ""
    for i in range(len(trajectory_list) // 3):
        s = trajectory_list[i * 3]
        a = trajectory_list[i * 3 + 1]
        r = trajectory_list[i * 3 + 2]

        hand_pos = str(list(policy._parse_obs(s)['hand_pos']))
        puck_pos = str(list(policy._parse_obs(s)['puck_pos']))
        goal_pos = str(list(policy._parse_obs(s)['goal_pos']))

        a = str(list(a))
        
        result += f"    {i + 1}. In time t_{i + 1}, you saw the following in an observation:\n"
        result += f"        - 'hand_pos' : {hand_pos}\n"
        result += f"        - 'puck_pos' : {puck_pos}\n"
        result += f"        - 'goal_pos' : {goal_pos}\n"
        result += f"      And, you did an action of : {a}, and you obtained {r} reward.\n\n"
    return result

def sublist_before_max(reward_list, traj_list):
    max_index = reward_list.index(max(reward_list))
    return traj_list[:(max_index + 1)*3 + 1]


import argparse

if __name__ == "__main__":

    t_init = time.time()

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--env_name', type=str, default='pick-place-v2')
    argparser.add_argument('--seed', type=int, default=111)
    argparser.add_argument('--task_idx', type=int, default=0)
    args = argparser.parse_args()


    SEED_NUMBER = args.seed
    task_idx = args.task_idx
    env_name = args.env_name
    ENV_NAME = ''.join(part.capitalize() for part in env_name.split('-'))

    system_prompt_path = "llm/system_prompt_RealMetaWorldPickPlaceV2_0411.txt"
    used_llm = 'gpt-3.5-turbo'


    env  = Make_Wrapped_Sawyer_Env_for_LLM(env_name, SEED_NUMBER, task_idx)
    done = False
    observation, info = env._reset()

    # Parse the configfile first
    config = parse_config('cfg.py')

    outdir = f"./outdir_{ENV_NAME}"
    # Create output directory if it does not already exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # make folder for npz file saving
    outdir_for_saving = f"./outdir_{ENV_NAME}/{ENV_NAME}"
    # Create output directory if it does not already exist
    if not os.path.exists(outdir_for_saving):
        os.makedirs(outdir_for_saving)    

    outdir_for_saving_ForRecordOnly = f"./outdir_{ENV_NAME}/{ENV_NAME}_ForRecordOnly"
    # Create output directory if it does not already exist
    if not os.path.exists(outdir_for_saving_ForRecordOnly):
        os.makedirs(outdir_for_saving_ForRecordOnly)

    # Set up logger
    log_file = os.path.join(outdir, f"logfile_{env_name}_seed{SEED_NUMBER}_idx{task_idx}.log")
    log_handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode='w')]

    logging.basicConfig(handlers=log_handlers, format=config['logging']['fmt'], level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    print(config['logging'])
    logger.setLevel(logging.getLevelName(config['logging']['level']))
    # logger.disabled = (not config['verbose'])
    logger.debug('boris is here in __main__ .')

    # set up OpenAI key
    config['api_key'] = "your_api_key_here"
    config['openai_org'] = "org-VFSaZQgJUGtyglfUzEzesHAC"
    
    
    # llm model initialisation in __main__ :
    init(config)

    # this b_intruction is added by myself. boris.
    b_instruction = (
        "Based on the previously asked questions and your answers, output a good 4-dim array in the form of [x, y, z, gripper_state].",
        "Based on the previously asked questions and your answers, render a proper 4-dim array in the style of [x, y, z, gripper_state].",
        "Based on the previously asked questions and your answers, supply a good 4-dim array in the form of [x, y, z, gripper_state].",
        "Based on the previously asked questions and your answers, indicate a good 4-dim array using [x, y, z, gripper_state] as the format.",
        "Based on the previously asked questions and your answers, come up with a good 4-dim array in the form of [x, y, z, gripper_state].",
        "Based on the previously asked questions and your answers, present a correct 4-dim array in the form of [x, y, z, gripper_state].",
    )
    selected_instruction = random.choice(b_instruction)

    # Templates for feedback
    r_feedback = (
        "Your reward for the latest step is {reward}.",
        "You got a reward of {reward}.",
        "The latest step brought you {reward} reward units.",
        "You've received a reward of {reward}.",
        "You've earned a reward of {reward}.",
        "You just got {reward} points.",
        "{reward} points for you.",
        "You've got yourself {reward} units of reward.",
        "The reward your latest step earned you is {reward}.",
        "The previous step's reward was {reward}.",
        "+{reward} reward",
        "Your reward is {reward}.",
        "The reward you just earned is {reward}.",
        "You have received {reward} points of reward.",
        "Your reward={reward}.",
        "The reward is {reward}.",
        "Alright, you just earned {reward} reward units.",
        "Your instantaneous reward is {reward}.",
        "Your rew. is {reward}.",
        "+{reward} points",
        "Your reward gain is {reward}."
    )


    
    global_traj_list_micro = [observation]
    global_traj_list_macro = [observation]
    global_traj_list_macro_for_LLM = [observation]
    global_step_count = []
    
    policy = env.mw_policy
    init_obs_info_dict = dict()
    init_obs_info_dict['hand_pos'] = str(list(policy._parse_obs(observation)['hand_pos']))
    init_obs_info_dict['puck_pos'] = str(list(policy._parse_obs(observation)['puck_pos']))
    init_obs_info_dict['goal_pos'] = str(list(policy._parse_obs(observation)['goal_pos']))

    succeeded = False


    while True:
        action = produce_initial_response(logger, init_obs_info_dict, str(selected_instruction), config['api_key'], gpt = used_llm, temperature = 0.3)
        if (action != []):
            break


    for _ in range(200):
        env.render()

    observation, reward, terminated, truncated, info = env._step(action)
    global_traj_list_micro.extend(info['traj_list_micro'])
    global_traj_list_macro.extend(info['traj_list_macro'])
    global_traj_list_macro_for_LLM.extend(info['traj_list_macro_for_LLM'])
    global_step_count.append(info['step_count'])
    
    if (info['success'] == True):
        succeeded = True

    # Choose a random feedback template
    chosen_template_rf = random.choice(r_feedback)
    
    # Plug in the reward value into the chosen template
    feedback_with_reward = chosen_template_rf.format(reward=reward)

    # terminated and truncated follow the same semantics as in Gymnasium
    done = terminated or truncated
    cnt = 0

    
    
    while ((not done) and (cnt <= 1000) and (not succeeded)):
        selected_instruction = random.choice(b_instruction)

        curr_obs_info_dict = dict()
        curr_obs_info_dict['hand_pos'] = str(list(policy._parse_obs(observation)['hand_pos']))
        curr_obs_info_dict['puck_pos'] = str(list(policy._parse_obs(observation)['puck_pos']))
        curr_obs_info_dict['goal_pos'] = str(list(policy._parse_obs(observation)['goal_pos']))

        # recursive past traj generation
        past_traj_string = generate_trajectory_string(global_traj_list_macro_for_LLM)
        
        
        action_valid = False
        while (not action_valid):
            further_action = produce_feedback_response(logger, curr_obs_info_dict, str(selected_instruction), str(feedback_with_reward), past_traj_string, config['api_key'], gpt = used_llm, temperature = 0.3)
            if (further_action != []):
                action_valid = True

        observation, reward, terminated, truncated, info = env._step(further_action)
        global_traj_list_micro.extend(info['traj_list_micro'])
        global_traj_list_macro.extend(info['traj_list_macro'])
        global_traj_list_macro_for_LLM.extend(info['traj_list_macro_for_LLM'])
        global_step_count.append(info['step_count'])
        
        # Choose a random feedback template
        chosen_template_rf = random.choice(r_feedback)
        
        # Plug in the reward value into the chosen template
        feedback_with_reward = chosen_template_rf.format(reward=reward)
    
        # terminated and truncated follow the same semantics as in Gymnasium
    
        done = terminated or truncated 

        cnt += 1
        
        if (info['success'] == True):
            succeeded = True


    print("done :", done)
    print("success :", info['success'])
    print("trucated:", truncated)
    print("terminated:", terminated)
    print("cnt:", cnt)
    


    # harvest our results
    all_s_micro, all_act_micro, all_r_micro = parse_traj_list(global_traj_list_micro)
    print("len(global_traj_list_micro) : ", len(global_traj_list_micro))
    print("len(all_s_micro) : ", len(all_s_micro))
    print("len(all_act_micro) : ", len(all_act_micro))
    print("len(all_r_micro) : ", len(all_r_micro))
    
    # further proceed and save npz only when max reward > 1 or succeeded
    if ((max(all_r_micro) >= 1) or info['success']):
        global_traj_list_micro_before_max = sublist_before_max(all_r_micro, global_traj_list_micro)
        all_s_micro_fore_max, all_act_micro_fore_max, all_r_micro_fore_max = parse_traj_list(global_traj_list_micro_before_max)
        # Saving micro before max
        np.savez(f'{outdir_for_saving}/{ENV_NAME}_4traj_seed{SEED_NUMBER}_idx{task_idx}_micro_before_max.npz', \
                all_traj=global_traj_list_micro_before_max, all_states=all_s_micro_fore_max, all_actions=all_act_micro_fore_max, all_rewards=all_r_micro_fore_max, last_info=info, global_step_count_list=global_step_count)

        # Saving micro in the same folder, 以防萬一
        np.savez(f'{outdir_for_saving}/{ENV_NAME}_4traj_seed{SEED_NUMBER}_idx{task_idx}_micro.npz', \
                 all_traj=global_traj_list_micro, all_states=all_s_micro, all_actions=all_act_micro, all_rewards=all_r_micro, last_info=info, global_step_count_list=global_step_count)

        print("Saved a before-max npz file.")
    else:
        # Saving micro in a different folder, should be only for recording purpose

        np.savez(f'{outdir_for_saving_ForRecordOnly}/{ENV_NAME}_4traj_seed{SEED_NUMBER}_idx{task_idx}_micro.npz', \
                 all_traj=global_traj_list_micro, all_states=all_s_micro, all_actions=all_act_micro, all_rewards=all_r_micro, last_info=info, global_step_count_list=global_step_count)

        print("Didn't save a before-max npz file. But did save a ForRecordingOnly npz file.")

    
    # Final time
    t_final = time.time()
    logger.debug('{} Program finished in {} secs.'.format(__name__, t_final - t_init))
    print('{} Program finished in {} secs.'.format(__name__, t_final - t_init))
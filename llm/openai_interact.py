import sys
sys.path.insert(0, ".")

from llm.openai_generate_response import openai_chat_response
import openai

def init(cfg):
    # Set up OpenAI api key
    openai.api_key = cfg['api_key']
    if cfg['openai_org'] is not None:
        openai.organization = cfg['openai_org'] 
    
    # with open(f'tasks/task_desc/{self.cfg["env_task"]}.txt', 'r') as file:
    #     task_desc = file.read()

    # Collect programs from LLM
    # programs = produce_program(self.logger, task_desc=task_desc, program_num=self.cfg['llm_program_num'], gpt=self.cfg['gpt'], temperature=self.cfg['llm_temperature'])

    # llm_log_dir = os.path.expanduser(os.path.join(self.cfg['outdir'], 'LLM'))
    # utils.cleanup_log_dir(llm_log_dir)

    # program_output_file = open(os.path.join(self.cfg['outdir'], 'LLM', 'programs.txt'), 'w')
    # llm_log_file = open(os.path.join(self.cfg['outdir'], 'LLM', 'log_programs.txt'), 'w')

def produce_program(logger, task_desc, program_num, init_state_str, gpt = "gpt-3.5-turbo", temperature = 0.7):
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

Please generate {program_num} unique Karel programs that could potentially solve the task. Ensure that each program is formatted on a single line and adheres to the correct Karel programming grammar. All programs should be formatted as `DEF run m( ... m)`."""

    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    response = openai_chat_response(messages=messages, temperature=temperature, gpt=gpt)

    logger.debug(f"System Prompt: {system_prompt}")
    logger.debug(f"User Prompt: {user_prompt}")
    logger.debug(f"LLM Response: {response}")

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

def produce_feedback_programs(logger, task_desc, program_num, init_state_str, trajectories_str, previous_programs_str, gpt = "gpt-3.5-turbo", temperature = 0.7):
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

Please generate {program_num} unique Karel programs that could potentially solve the task. Ensure that each program is formatted on a single line and adheres to the correct Karel programming grammar.\n\n"""

    # Previous programs

    user_prompt += "The response should be format as follows: \n"
    user_prompt += "REFLACTION: <your reflection on the agent's performance>\n"
    user_prompt += "PROGRAMS:\n1. <your program>\n2. <your program>\n3. <your program>\n...\n"

    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    response = openai_chat_response(messages=messages, temperature=temperature, gpt=gpt)

    logger.debug(f"System Prompt: {system_prompt}")
    logger.debug(f"User Prompt: {user_prompt}")
    logger.debug(f"LLM Response: {response}")

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
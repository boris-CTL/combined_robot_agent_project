import time
import argparse
import os
import logging
import importlib.util
import sys

from tensorboardX import SummaryWriter
import wandb
from wandb.integration.sb3 import WandbCallback
import torch
import gymnasium as gym
from stable_baselines3 import A2C, PPO, DQN
from karel_env.karel_env import KarelWorldEnv
# from karel_env.program_env import KarelProgramEnv
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import random
import pickle
import shutil
import imageio
from PIL import Image
from pygifsicle import optimize
from karel_env.dsl.dsl_parse import parse
from karel_env.karel_world import Karel_world, KarelStateGenerator
import tqdm as tqml
from llm.llama_interact import init as llm_init
from llm.llama_interact import produce_program as llm_produce_program
from llm.llama_interact import produce_feedback_programs as llm_produce_feedback_programs

def save_gif(karel_world, path, s_h):
    # create video
    frames = []
    for s in s_h:
        frames.append(np.uint8(karel_world.state2image(s=s).squeeze()))
    frames = np.stack(frames, axis=0)
    imageio.mimsave(path, frames, format='GIF-PIL', fps=5)
    #optimize(path)
    return


from torch.utils.data.dataset import Dataset, random_split
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions

    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len(self.observations)

def pretrain_agent(
    student,
    train_expert_dataset,
    env,
    batch_size=64,
    epochs=1000,
    scheduler_gamma=0.7,
    learning_rate=1.0,
    log_interval=100,
    no_cuda=True,
    seed=1,
    test_batch_size=64,
):
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device: {}".format(device))
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if isinstance(env.action_space, gym.spaces.Box):
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Extract initial policy
    model = student.policy.to(device)

    def train(model, device, train_loader, optimizer):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            if isinstance(env.action_space, gym.spaces.Box):
                # A2C/PPO policy outputs actions, values, log_prob
                # SAC/TD3 policy outputs actions only
                if isinstance(student, (A2C, PPO)):
                    action, _, _ = model(data)
                else:
                    # SAC/TD3:
                    action = model(data)
                action_prediction = action.double()
            else:
                # Retrieve the logits for A2C/PPO when using discrete actions
                dist = model.get_distribution(data)
                action_prediction = dist.distribution.logits
                target = target.long()

            loss = criterion(action_prediction, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

    # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training
    # and testing
    train_loader = torch.utils.data.DataLoader(
        dataset=train_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )

    # Define an Optimizer and a learning rate schedule.
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    # Now we are finally ready to train the policy model.
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer)
        scheduler.step()

    # Implant the trained policy network back into the RL student agent
    student.policy = model

def collect_program_rollouts(config, logger, model, llm_model, trajectories_str, program_karel_world, program_state_generator, reflection=False, previous_programs_str="", baseline=0.0):
    """Collect rollouts for programs"""
    # return

    with open(f'tasks/task_desc/{config["env_task"]}.txt', 'r') as file:
        task_desc = file.read()

    # create log directory for programs
    if not os.path.exists(os.path.join(config['outdir'], 'programs')):
        os.makedirs(os.path.join(config['outdir'], 'programs'))
    
    # create program log file
    program_log_file = os.path.join(config['outdir'], 'programs', 'programs.log')

    init_func = eval(f'program_state_generator.generate_single_state_{config["env_task"]}')

    init_state, _, _, _, metadata = init_func(config['input_height'], config['input_width'], config['wall_prob'])
    init_state_str = program_karel_world.state2str(init_state)

    if 'model_name' == "None":
        logger.warning('Model name is not provided, GT programs will be used')
        # Here programs are manually written
        programs = ['DEF run m( WHILE c( markersPresent c) w( WHILE c( markersPresent c) w( pickMarker move w) turnRight move turnLeft WHILE c( markersPresent c) w( pickMarker move w) turnLeft move turnRight w) m)', 'DEF run m( WHILE c( markersPresent c) w( WHILE c( markersPresent c) w( pickMarker move w) turnRight move turnLeft WHILE c( markersPresent c) w( pickMarker move w) turnLeft move turnRight w) m)', 'DEF run m( WHILE c( markersPresent c) w( WHILE c( markersPresent c) w( pickMarker move w) turnRight move turnLeft WHILE c( markersPresent c) w( pickMarker move w) turnLeft move turnRight w) m)']
    # Generate programs
    elif reflection:
        assert trajectories_str is not None
        programs = llm_produce_feedback_programs(logger, task_desc=task_desc, program_num=config['llm_program_num'], init_state_str=init_state_str, trajectories_str=trajectories_str, previous_programs_str=previous_programs_str, llm=llm_model)
    else:
        programs = llm_produce_program(logger, task_desc=task_desc, program_num=config['llm_program_num'], init_state_str=init_state_str, llm=llm_model)

    PREVIOUS_PROGRAMS_STR = ""
    # num_demo = model.env.num_envs
    num_demo = config['num_demo_per_program']
    logger.debug('Number of demos: {}'.format(num_demo))
    collect_count = 0
    rewards_list = []
    expert_observations = []
    expert_actions = []
    for program_str in programs:
        if program_str[:4] != 'DEF ' or program_str[-2:] != 'm)':
            logger.warning('Program {} is not valid'.format(program_str))
            continue

        PREVIOUS_PROGRAMS_STR += "Program: " + program_str + "\n"
        
        exe, s_exe = parse(program_str, environment='karel')
        if not s_exe or not len(program_str) > 4:
            logger.warning('Program {} is not valid'.format(program_str))
        else:
            rewards = 0.0
            # write program to log file
            with open(program_log_file, 'a') as f:
                f.write('Program: {}\n'.format(program_str))
            logger.debug('Program: {}'.format(program_str))

            for i in range(num_demo):

                # get initial state
                init_state, _, _, _, metadata = init_func(config['input_height'], config['input_width'], config['wall_prob'])
                
                program_karel_world.set_new_state(init_state, metadata)

                exe, s_exe = parse(program_str)
                if not s_exe:
                    raise RuntimeError('This should be correct')

                program_karel_world, n, s_run = exe(program_karel_world, 0)

                # we expect to return execution traces in (input, ...., output) format for EGPS
                # if no actions were executed in environment, repeat initial state and add dummy action for it
                if len(program_karel_world.a_h) < 1:
                    assert len(program_karel_world.s_h) == 1
                    program_karel_world.a_h.append(program_karel_world.num_actions)
                    program_karel_world.s_h.append(program_karel_world.s_h[0])
                    program_karel_world.r_h.append(0.0)

                s_h_list = np.stack(program_karel_world.s_h, axis=0)
                a_h_list = np.array(program_karel_world.a_h)
                r_h_list = np.array(program_karel_world.r_h)
                done_h_list = [False for _ in range(len(s_h_list))]
                done_h_list[-1] = program_karel_world.done
                episode_reward = np.sum(r_h_list)

                logger.debug('Sample {} : {}'.format(i, episode_reward))
                rewards += float(episode_reward)
                
                # write to log file
                with open(program_log_file, 'a') as f:
                    f.write('Sample {} : {}\n'.format(i, episode_reward))
                    # write length of s_h, a_h, r_h
                    f.write('Length of s_h: {}\n'.format(len(s_h_list)))
                    f.write('Length of a_h: {}\n'.format(len(a_h_list)))
                    f.write('Length of r_h: {}\n'.format(len(r_h_list)))
                    # write s_h, a_h, r_h
                    f.write('s_h: {}\n'.format(s_h_list))
                    f.write('a_h: {}\n'.format(a_h_list))
                    f.write('r_h: {}\n\n'.format(r_h_list))

                if i == 0:
                    save_video_path = os.path.join(config['outdir'], 'programs', '{}_sample{}.gif'.format(config['env_task'], i))
                    save_gif(program_karel_world, save_video_path, s_h_list)

                # if episode reward > 0, collect the episode
                if episode_reward > baseline:
                    # collect rollouts for each transition
                    info = [{"TimeLimit.truncated": False}]
                    for j in range(len(a_h_list)):
                        # add transition to replay buffer
                        # model.replay_buffer.add(s_h_list[j], s_h_list[j+1], a_h_list[j], r_h_list[j], done_h_list[j], info)
                        expert_observations.append(s_h_list[j])
                        expert_actions.append(a_h_list[j])
                    collect_count += 1
                
                program_karel_world.clear_history()

            logger.debug('Average : {}'.format(rewards/num_demo))
            rewards_list.append(rewards/num_demo)
            PREVIOUS_PROGRAMS_STR += "Episodic reward: " + str(rewards/num_demo) + "\n\n"

    if config['offline_timesteps'] <= 0:
        return programs, rewards_list, PREVIOUS_PROGRAMS_STR

    if collect_count <= 0:
        # input('Collect count is 0, press any key to continue')
        logger.warning('Collect count is 0')
        return programs, rewards_list, PREVIOUS_PROGRAMS_STR

    expert_observations = np.array(expert_observations)
    expert_actions = np.array(expert_actions)

    expert_dataset = ExpertDataSet(expert_observations, expert_actions)

    pretrain_agent(
        model, 
        expert_dataset,
        model.get_env(),
        epochs=config['offline_timesteps'],
        no_cuda=False,
        seed=config['seed'],
    )

    # callback.on_training_end()

    return programs, rewards_list, PREVIOUS_PROGRAMS_STR


def run(config, logger):

    if config['logging']['wandb']:
        import wandb
        wandb.init(project="llm-prl", sync_tensorboard=True, id=config['outdir'].split('/')[-1])
    else:
        os.environ['WANDB_MODE'] = 'dryrun'

    # begin block: this block sets the device from the config
    if config['device'].startswith('cuda') and torch.cuda.is_available():
        device = torch.device(config['device'])
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        logger.warning('{} GPU not available, running on CPU'.format(__name__))

    # setup tensorboardX: create a summary writer
    writer = SummaryWriter(logdir=config['outdir'])

    # this line logs the device info
    logger.debug('{} Using device: {}'.format(__name__, device))

    # end block: this block looks good

    # begin block: this block sets random seed for the all the modules
    if config['seed'] is not None:
        logger.debug('{} Setting random seed'.format(__name__))
        seed = config['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

    if config['device'].startswith('cuda') and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # end block: this block looks good. if we have specified a seed, then we set it

    global_logs = {'info': {}, 'result': {}}

    # make dummy env to get action and observation space based on the environment
    custom_kwargs = {"config": config, "env_task": config['env_task']}
    # custom = True if "karel" or "CartPoleDiscrete" in config['env_name'] else False
    
    # Add wandb logger to the model
    if config['logging']['wandb']:
        wandb.config.update(config)

    # Save configs and models
    pickle.dump(config, file=open(os.path.join(config['outdir'], 'config.pkl'), 'wb'))
    shutil.copy(src=config['configfile'], dst=os.path.join(config['outdir'], 'configfile.py'))

    # begin block: this block creates the environment
    # if custom:
    gym.register(id="KarelWorld", entry_point=KarelWorldEnv, kwargs=custom_kwargs)
    env = gym.make("KarelWorld", **custom_kwargs)
    logger.debug('Using environment: KarelWorld')
    logger.debug('Env task: {}'.format(config['env_task']))

    # end block: this block looks good

    # begin block: this block creates the model
    logger.debug('Using algorithm: {}'.format(config['algorithm']))
    model = eval(config['algorithm'])(
        policy=config['policy'], env=env, 
        verbose=1, tensorboard_log=config['outdir'], 
        seed=config['seed'], device=device)
    # end block: this block looks good

    karel_world = Karel_world(make_error=config['make_error'], env_task=config['env_task'])
    state_generator = KarelStateGenerator(seed=config['seed'])

    # Creates the programs
    if config['collect_program_rollouts']:
        logger.debug('Creating programs')
        if 'model_name' != "None":
            llm_model = llm_init(config)
        # programs = llm_produce_program(logger, task_desc=task_desc, program_num=config['llm_program_num'], gpt=config['gpt'], temperature=config['llm_temperature'])
        # # read programs from file
        # with open(config['programs_file'], 'r') as f:
        #     for line in f:
        #         programs.append(line.strip())

    # begin block: this block collects the rollouts
    if config['collect_program_rollouts']:
        logger.debug('Collecting program rollouts')
        programs, programs_reward_list, previous_programs_str = collect_program_rollouts(config, logger, model, llm_model, None, karel_world, state_generator, reflection=False)
        writer.add_text('programs', str(programs), 0)
        writer.add_scalar('programs_mean_reward', np.mean(programs_reward_list), 0)
        writer.add_scalar('programs_max_reward', np.max(programs_reward_list), 0)
        writer.add_scalar('programs_min_reward', np.min(programs_reward_list), 0)
    
    logger.debug('Evaluating model')
    eval_results = evaluate_policy(model, env, n_eval_episodes=config['n_eval_episodes'], deterministic=False, render=True, callback=None, reward_threshold=None, return_episode_rewards=False)
    logger.debug('Logging results')
    logger.info('Eval results:\n  Mean reward: {}\n  Std of reward: {}'.format(eval_results[0], eval_results[1]))
    global_logs['result'] = eval_results
    writer.add_scalar('eval_mean_reward', eval_results[0], 0)
    # end block: this block looks good

    action_mapping = {
        0: 'move',
        1: 'turnLeft',
        2: 'turnRight',
        3: 'pickMarker',
        4: 'putMarker'
    }

    logger.debug('Training model')
    epoch = 10
    for i in range(epoch):
        # train the model
        model.learn(total_timesteps=config['total_timesteps'], reset_num_timesteps=False, callback=WandbCallback())

        logger.debug('Evaluating model')
        eval_results = evaluate_policy(model, env, n_eval_episodes=config['n_eval_episodes'], deterministic=False, render=False, callback=None, reward_threshold=None, return_episode_rewards=False)

        logger.debug('Logging results')
        logger.info('Eval results:\n  Mean reward: {}\n  Std of reward: {}'.format(eval_results[0], eval_results[1]))
        global_logs['result'] = eval_results
        writer.add_scalar('eval_mean_reward', eval_results[0], i+1)

        if not config['collect_program_rollouts']:
            continue
            
        # sample some trajectories from the policy
        logger.debug('Collecting trajectory from policy')
        trajectories = ""
        trajectories += "The trajectories will be shown in the format of: Initial state -> Action -> Reward -> Action -> Reward -> ... -> Last state -> TERMINATED/TRUNCATED\n\n"
    
        # Log 10 episode trajectories
        episode_rewards = []
        for j in range(3): 
            episode_reward = 0
            obs, info = env.reset()
            trajectory = []
            while True:
                action, _states = model.predict(obs, deterministic=False)
                new_obs, rewards, terminated, truncated, info = env.step(action)
                trajectory.append((obs, action, rewards, new_obs, terminated, truncated, info))
                episode_reward += rewards
                obs = new_obs
                if terminated or truncated:
                    break
            episode_rewards.append(episode_reward)
            # transfer trajectory to string description for LLM understanding
            trajectory_str = ''
            # enumerate trajectory
            trajectory_str += 'Initial state: \n'
            trajectory_str += karel_world.state2str(trajectory[0][0])
            for idx, (obs, action, rewards, new_obs, terminated, truncated, info) in enumerate(trajectory):
                # trajectory_str += f'State {idx}: \n'
                # trajectory_str += karel_world.state2str(obs)
                # trajectory_str += '\nAction: '
                trajectory_str += action_mapping[int(action)]
                trajectory_str += ' '
                trajectory_str += str(rewards)
                trajectory_str += '\n'
                if terminated:
                    trajectory_str += 'Last state: \n'
                    trajectory_str += karel_world.state2str(new_obs)
                    trajectory_str += 'TERMINATED\n\n'
                elif truncated:
                    trajectory_str += 'Last state: \n'
                    trajectory_str += karel_world.state2str(new_obs)
                    trajectory_str += 'TRUNCATED\n\n'
                
            trajectories += "Trajectory {}:\n".format(j+1)
            trajectories += trajectory_str

        writer.add_text('trajectories', trajectories, i)

        mean_reward = np.mean(episode_rewards)
        writer.add_scalar('trajectories_mean_reward', mean_reward, i+1)

        # produce programs from trajectories
        logger.debug('Producing programs from trajectories')
        # programs = llm_produce_feedback_programs(logger, task_desc=task_desc, program_num=config['llm_program_num'], trajectories_str=trajectories, gpt=config['gpt'], temperature=config['llm_temperature'])

        # collect rollouts for each program
        logger.debug('Collecting rollouts for each program')
        programs, programs_reward_list, previous_programs_str = collect_program_rollouts(config, logger, model, llm_model, trajectories, karel_world, state_generator, reflection=True, previous_programs_str=previous_programs_str,  baseline=eval_results[0])
        writer.add_text('programs', str(programs), i+1)
        writer.add_scalar('programs_mean_reward', np.mean(programs_reward_list), i+1)
        writer.add_scalar('programs_max_reward', np.max(programs_reward_list), i+1)
        writer.add_scalar('programs_min_reward', np.min(programs_reward_list), i+1)


    # train the model after LLM loop
    model.learn(total_timesteps=3e6, reset_num_timesteps=False, callback=WandbCallback())

    # begin block: this block saves the model
    logger.debug('Saving model')
    model.save(os.path.join(config['outdir'], 'model'))
    # save replay buffer
    if config['algorithm'] == 'DQN':
        model.save_replay_buffer(os.path.join(config['outdir'], 'replay_buffer.pkl'))
    # end block: this block looks good

    # begin block: this block saves the logs
    logger.debug('Saving logs')
    pickle.dump(global_logs, file=open(os.path.join(config['outdir'], 'logs.pkl'), 'wb'))
    # end block: this block looks good

    # begin block: this block closes the environment
    logger.debug('Closing environment')
    env.close()
    # end block: this block looks good

    # begin block: this block closes the summary writer
    logger.debug('Closing summary writer')
    writer.close()
    # end block: this block looks good

    # begin block: this block closes wandb
    if config['logging']['wandb']:
        logger.debug('Closing wandb')
        wandb.finish()
    # end block: this block looks good

    return global_logs


def parse_config(configfile):
    spec = importlib.util.spec_from_file_location('cfg', configfile)
    conf_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf_mod)
    config = conf_mod.config
    print(config)
    return config

if __name__ == "__main__":

    t_init = time.time()
    parser = argparse.ArgumentParser(description='Run LLMPRL')

    # Add arguments (including a --configfile)
    parser.add_argument('-o', '--outdir',
                        help='Output directory for results', default='pretrain/output_dir')
    parser.add_argument('-d', '--datadir',
                        help='dataset directory containing data.hdf5 and id.txt')
    parser.add_argument('-c', '--configfile',
                        help='Input file for parameters, constants and initial settings')
    parser.add_argument('-v', '--verbose',
                        help='Increase output verbosity', action='store_true')
    parser.add_argument('-p', '--prefix', default='test',
                        help='Prefix for output directory')
    parser.add_argument('-s', '--seed', default=0, type=int,
                        help='Random seed')
    parser.add_argument('-a', '--algorithm', default='DQN',
                        help='RL algorithm')
    parser.add_argument('--env_task', default='harvester', type=str, help='task name')
    parser.add_argument('--collect_program_rollouts', action='store_true', help='collect program rollouts')
    parser.add_argument('--feedback_programs', action='store_true', help='collect feedback programs')
    parser.add_argument('--offline_timesteps', default=0, type=str, help='number of offline timesteps')
    parser.add_argument('--programs_file', default='programs.txt', type=str, help='file containing programs')
    parser.add_argument('--total_timesteps', default=10e6, type=str, help='number of total timesteps')
    parser.add_argument('--api_key', type=str, help='OpenAI API key')
    parser.add_argument('--openai_org', type=str, help='OpenAI organization')
    parser.add_argument('--num_demo_per_program', default=10, type=int, help='number of demos per program')
    parser.add_argument('--input_height', default=8, type=int, help='input height')
    parser.add_argument('--input_width', default=8, type=int, help='input width')

    # Parse arguments
    args = parser.parse_args()

    args.total_timesteps = int(float(args.total_timesteps))
    args.offline_timesteps = int(float(args.offline_timesteps))

    # Parse the configfile first
    config = parse_config(args.configfile)
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v

    config['outdir'] = os.path.join(args.outdir, config['env_task'], '%s-offline%s-%s-%s-%s-%s' % (config['env_task'], config['offline_timesteps'], config['algorithm'], 'spiking' if config['collect_program_rollouts'] else 'scratch', config['prefix'], config['seed'] ))

    # Create output directory if it does not already exist
    if not os.path.exists(config['outdir']):
        os.makedirs(config['outdir'])

    # Set up logger
    log_file = os.path.join(config['outdir'], config['logging']['log_file'])
    log_handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode='w')]
    logging.basicConfig(handlers=log_handlers, format=config['logging']['fmt'], level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    print(config['logging'])
    logger.setLevel(logging.getLevelName(config['logging']['level']))
    logger.disabled = (not config['verbose'])

    # Call the main method
    run_results = run(config, logger)

    # Final time
    t_final = time.time()
    logger.debug('{} Program finished in {} secs.'.format(__name__, t_final - t_init))
    print('{} Program finished in {} secs.'.format(__name__, t_final - t_init))

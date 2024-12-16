import torch
import random
from stable_baselines3 import PPO, A2C  # Assuming you're using Stable Baselines3 or a similar library
from MetaWorldEnv_micro import MetaWorldMultiTaskEnv

import wandb
from wandb.integration.sb3 import WandbCallback

import click
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import gymnasium as gym
import numpy as np
from torch.utils.data.dataset import Dataset

# import imageio
import imageio.v2 as iio


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
    no_cuda=False,
    seed=42,
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

@click.command()
@click.option('--seed', default=42)
# @click.option('--total_steps', default=1e6)
@click.option('--epochs', default=4000)
@click.option('--batch_size', default=5000) # not minibatch size
@click.option('--n_tasks', default=50)
@click.option('--env_name', default="assembly-v2")
@click.option('--eval_interval', default=20)
@click.option('--visual_interval', default=100)
@click.option('--do_pretrain', default=False)
@click.option('--pretrain_dataset', default=None)
@click.option('--pretrain_trajs', default=None)
@click.option('--postfix', default="_0321_steps1e6_with_pretrain")
@click.option('--log_wandb', default=True)

def train_agent(seed, epochs, batch_size, n_tasks, env_name, eval_interval, visual_interval, do_pretrain, pretrain_dataset, pretrain_trajs, postfix, log_wandb):
    # Check if model is already trained
    if os.path.exists(f"output/ppo_{env_name}{postfix}/ckpt/ppo_{env_name}{postfix}.zip"):
        print(f"Model already trained for {env_name}.")
        return

    my_config = {
        "env_name": env_name,
        # "total_steps": total_steps,
        "epochs": epochs,
        "batch_size": batch_size,
        "n_tasks": n_tasks,
        "seed": seed,
        "eval_interval": eval_interval,
        "visual_interval": visual_interval,
        "do_pretrain": do_pretrain,
        "pretrain_dataset": pretrain_dataset,
        "pretrain_trajs": pretrain_trajs,
        "log_wandb": log_wandb,
        "run_id": "ppo_" + env_name + postfix,
        "eval_episode_num": 10,
    }

    print(my_config)

    # Create wandb session (Uncomment to enable wandb logging)
    if log_wandb:
        run = wandb.init(
            project="RLLM",
            config=my_config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            id=my_config["run_id"]
        )

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)

    # Initialize Meta-World
    env = MetaWorldMultiTaskEnv(env_name, n_tasks, seed=seed)
    env.seed(seed)
    # due to the implementation of MetaWorldMultiTaskEnv, need to reset() first, calling pure_reset()
    # obs_, info_ = env.pure_reset()

    # Initialize PPO Agent
    model = PPO("MlpPolicy", 
                env, 
                learning_rate=5e-4,
                verbose=1, 
                tensorboard_log=f"tensorboard_logs/ppo_{env_name}{postfix}",
                n_steps=5000,
                n_epochs=epochs,
                batch_size=50, # change 32 to 50
                seed=seed,
                gamma=0.99,
                device="cuda",
                )

    # create output folder
    os.makedirs(f'output/{my_config["run_id"]}', exist_ok=True)
    os.makedirs(f'output/{my_config["run_id"]}/ckpt', exist_ok=True)

    # create csv file to store evaluation results
    # key: global_step, num_traj, avg_score, avg_success, std_score, std_success
    eval_results = open(f'output/{my_config["run_id"]}/eval_results.csv', 'a')
    eval_results.write("global_step,num_traj,avg_score,avg_success,std_score,std_success\n")

    def eval(env, model, epoch=None):
        print("--------------------Evaluation--------------------")
        success_list = []
        reward_list = []
        for seed in range(my_config["eval_episode_num"]):
            done = False

            # Set seed using old Gym API
            env.seed(seed)
            obs, info = env.reset()

            terminated = False
            truncated = False
            successed = False

            frames = []

            if epoch == 'pretrained' or epoch % visual_interval == 0:
                # create output folder
                os.makedirs(f'output/{my_config["run_id"]}/epoch{epoch}', exist_ok=True)
                # create video writer
                writer = iio.get_writer(f'output/{my_config["run_id"]}/epoch{epoch}/sample{seed+1}.mp4', format='FFMPEG', mode='I', fps=80)

                # save rgb_array to video
                writer.append_data(env.render(mode = "rgb_array"))

            # Interact with env using old Gym API
            while not terminated and not truncated:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                successed = info['success']

                if epoch == 'pretrained' or epoch % visual_interval == 0:
                    writer.append_data(env.render(mode = "rgb_array"))
            
            if epoch == 'pretrained' or epoch % visual_interval == 0:
                # close video writer
                writer.close()
                print(f"\tVideo saved for episode {seed+1}.")
            
            print(f"\tEvaluation episode {seed+1} done.\n\t Success: {successed}\n\t Reward: {reward}")
            
            success_list.append(successed)
            reward_list.append(reward)

        if epoch is not None:
            print("Epoch: ", epoch)

        eval_dict = {
            "avg_score": np.mean(reward_list),
            "avg_success": np.mean(success_list),
            "std_score": np.std(reward_list),
            "std_success": np.std(success_list)
        }
        
        print(f"Avg_score:  {eval_dict['avg_score']} +/- {eval_dict['std_score']}")
        print(f"Avg_success:  {eval_dict['avg_success']} +/- {eval_dict['std_success']}")
        print()
        return eval_dict

    max_avg_score = 0

    eval_dict = eval(env, model, 0)
    if log_wandb:
        wandb.log({"avg_score": eval_dict["avg_score"], "avg_success": eval_dict["avg_success"]}, step=0)

    # Record initial evaluation
    eval_results.write(f"{0},{0},{eval_dict['avg_score']},{eval_dict['avg_success']},{eval_dict['std_score']},{eval_dict['std_success']}\n")

    pretrain_steps = 0
    pretrain_trajs = 0

    # Pretrain
    if do_pretrain:
        assert pretrain_dataset is not None, "Please provide a pretrain dataset"
        assert pretrain_trajs is not None, "Please provide the number of pretrain trajs"
        
        print("Loading pretrain dataset from", pretrain_dataset)
        # Load .npz file
        pretrained_data = np.load(pretrain_dataset, allow_pickle=True)

        pretrain_dataset = ExpertDataSet(pretrained_data['expert_observations'], pretrained_data['expert_actions'])

        pretrain_steps = len(pretrain_dataset)
        pretrain_trajs = int(pretrain_trajs)
        print(f"Dataset loaded. {pretrain_steps} samples found.")
        print("Pretraining...")
        
        pretrain_agent(model,
                    pretrain_dataset,
                    env,
                    batch_size=512,
                    epochs=100,
                    scheduler_gamma=0.7,
                    learning_rate=1.0,
                    log_interval=100,
                    no_cuda=False,
                    seed=seed)
        
        print("Pretraining done.")

        # Save Model
        model.save(f"output/{my_config['run_id']}/ckpt/ppo_{env_name}{postfix}_pretrained")
        print("Pretrained model saved.")

        eval_dict = eval(env, model, 'pretrained')
        if log_wandb:
            wandb.log({"avg_score": eval_dict["avg_score"], "avg_success": eval_dict["avg_success"]}, step=pretrain_steps)

        # Record initial evaluation
        eval_results.write(f"{pretrain_steps},{pretrain_trajs},{eval_dict['avg_score']},{eval_dict['avg_success']},{eval_dict['std_score']},{eval_dict['std_success']}\n")


    # Train
    for epoch in range(epochs):
        env.reset()
        if log_wandb:
            model.learn(total_timesteps=batch_size,
                        reset_num_timesteps=False,
                        callback=WandbCallback(
                            gradient_save_freq=100,
                            verbose=2,
                        ))
        else:
            model.learn(total_timesteps=batch_size,
                        reset_num_timesteps=False)

        # Evaluate
        if (epoch+1) % eval_interval == 0:
            eval_dict = eval(env, model, epoch+1)
            if log_wandb:
                wandb.log({"avg_score": eval_dict["avg_score"], "avg_success": eval_dict["avg_success"]}, step=pretrain_steps+batch_size*(epoch+1))
            # Record evaluation results
            eval_results.write(f"{pretrain_steps+batch_size*(epoch+1)},{pretrain_trajs+(batch_size*(epoch+1))//500},{eval_dict['avg_score']},{eval_dict['avg_success']},{eval_dict['std_score']},{eval_dict['std_success']}\n")

            if eval_dict["avg_score"] >= max_avg_score:
                max_avg_score = eval_dict["avg_score"]
                model.save(f"output/{my_config['run_id']}/ckpt/ppo_{env_name}{postfix}_best")
                print(f"Model saved as best model at epoch {epoch+1}")
        

    # Save Model
    model.save(f"output/{my_config['run_id']}/ckpt/ppo_{env_name}{postfix}")

    # Close wandb session
    if log_wandb:
        run.finish()
    
    # Close evaluation results file
    eval_results.close()

train_agent()
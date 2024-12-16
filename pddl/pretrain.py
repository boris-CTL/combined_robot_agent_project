def collect_sa_rollouts(config, env, logger, model):
    actions = np.array([])
    observations = np.array([])
    programs = []
    if config['grammar_type']=='file':
        # read programs from file
        with open(config['programs_file'], 'r') as f:
            for line in f:
                programs.append(line.strip())
    elif config['grammar_type']=='program_random':
        for i in range(config['num_programs']):
            programs.append(generate_harvester_program())
    elif config['grammar_type']=='heuristic':
        for i in range(config['num_programs']):
            done_collect = False
            while not done_collect:              
                obs, _ = env.reset()
                acts = np.array([])
                obss = np.array([])
                done = False
                frames = []
                while not done:
                    action = policy(obs)
                    obss = np.append(obss, obs)
                    acts = np.append(acts, action)
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    done = terminated or truncated
                    frame = env.render()
                    frames.append(Image.fromarray(frame))
                    # env.render()
                    # save the trajectory
                    if terminated or truncated or i == config['max_episode_steps'] - 1:
                        save_video_path = os.path.join(config['outdir'], 'rollouts')
                        if not os.path.exists(save_video_path):
                            os.makedirs(save_video_path)
                            '{}_rollout.gif'.format(config['env_task'])
                        save_video_name = '{}_rollout.gif'.format(config['env_task'])
                        save_frame_gif(save_video_path, save_video_name, frames)
                        print("done", len(obss))
                        if len(obss) < 360:
                            done_collect = True
                            break
            observations = np.append(observations, obss)
            actions = np.append(actions, acts)
    return observations, actions

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
    env,
    train_expert_dataset,
    test_expert_dataset,
    batch_size=64,
    epochs=1000,
    scheduler_gamma=0.7,
    learning_rate=1.0,
    log_interval=100,
    no_cuda=True,
    seed=1,
    test_batch_size=64,
):
    def train(model, device, train_loader, optimizer):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = {t: data[t].to(device) for t in data}, target.to(device)
            optimizer.zero_grad()

            if isinstance(env.action_space, spaces.Box):
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

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = {t: data[t].to(device) for t in data}, target.to(device)
                if isinstance(env.action_space, spaces.Box):
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

                test_loss = criterion(action_prediction, target)
        test_loss /= len(test_loader.dataset)
        print(f"Test set: Average loss: {test_loss:.4f}")

    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if isinstance(env.action_space, spaces.Box):
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Extract initial policy
    model = student.policy.to(device)

    # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training
    # and testing
    train_loader = torch.utils.data.DataLoader(
        dataset=train_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_expert_dataset,
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs,
    )

    # Define an Optimizer and a learning rate schedule.
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)
    # Now we are finally ready to train the policy model.
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        scheduler.step()

    # Implant the trained policy network back into the RL student agent
    student.policy = model

    return student

def BC_pretrain(config, env, logger, model):
    """Collect rollouts for programs"""
    # return
    observations, actions = collect_sa_rollouts(config, env, logger, model)

    expert_observations = observations
    expert_actions = actions

    expert_dataset = ExpertDataSet(expert_observations, expert_actions)

    train_size = int(0.8 * len(expert_dataset))

    test_size = len(expert_dataset) - train_size

    train_expert_dataset, test_expert_dataset = random_split(
        expert_dataset, [train_size, test_size]
    )

    model = pretrain_agent(
        model,
        env,
        train_expert_dataset,
        test_expert_dataset,
        epochs=config['pretrain_epochs'],
        scheduler_gamma=0.7,
        learning_rate=1.0,
        log_interval=100,
        no_cuda=True,
        seed=1,
        batch_size=64,
        test_batch_size=1000,
    )

    return model

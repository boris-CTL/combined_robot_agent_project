import gymnasium as gym
from gymnasium import spaces
from karel_env.karel_world import Karel_world, KarelStateGenerator
import numpy as np

class KarelWorldEnv(gym.Env):
    def __init__(self, config, env_task):
        super(KarelWorldEnv, self).__init__()
        self.metadata = {'render.modes': ['rgb_array', 'init_states']}
        self.config = config
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._max_episode_steps = config['max_episode_steps']

        print("KarelWorldEnv_max_episode_steps: ", self._max_episode_steps)

        self.karel_world = Karel_world(make_error=config['make_error'], env_task=env_task)
        self.state_generator = KarelStateGenerator(seed=config['seed'])
        self.init_func = eval(f'self.state_generator.generate_single_state_{env_task.replace("_sparse", "")}')
        self.init_state, _, _, _, self.metadata = self.init_func(config['input_height'], config['input_width'], config['wall_prob'])
        self.karel_world.set_new_state(self.init_state, self.metadata)
        
        # define action space
        self.action_space = spaces.Discrete(5)

        # define observation space
        if config['obv_type'] == "state":
            self.initial_obv = self.init_state.copy()
            self.observation_space = spaces.Box(low=0, high=1, shape=(self.initial_obv.shape), dtype=np.bool_)
            print("KarelWorldEnv_obv_shape: ", self.initial_obv.shape)
        else:
            raise NotImplementedError('observation not recognized')

    def step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
            The action taken by the agent in the environment.
        Returns
        -------
        state : numpy.array
            The state of the environment after the agent has stepped through.
        reward : float
            The reward for stepping through the environment.
        terminated : bool
            Whether the episode has terminated.
        truncated : bool
            Whether the episode has truncated.
        info : dict
            Additional information about the environment.
        """

        info = {}

        # Execute action
        self.karel_world.state_transition(action)
        reward = self.karel_world.r_h[-1]
        done = self.karel_world.done
        self._episode_steps += 1
        self._episode_reward += reward

        # Create state image
        self.state = self.karel_world.s.copy()

        # Log info about the state
        # info['r'] = reward
        # info['a'] = action
        # info['s'] = self.state
        # info['done'] = self.karel_world.done

        self.a_h.append(action)
        self.s_image_h.append(self.state.copy())

        # Check if the episode is done
        done = self.karel_world.done
        truncated = self._episode_steps >= self._max_episode_steps

        # Log info about the episode
        if done or truncated:
            info['TimeLimit.truncated'] = not done
            info['episode'] = {'r': self._episode_reward, 'a': self.a_h, 's': self.s_image_h, 'l': self._episode_steps}

        # Return the next state, reward, done, truncated, and info
        return self.state, reward, done, truncated, info
       

    def reset(self, seed=None):
        # Reset the state of the environment to an initial state
        self._elapsed_steps = 0
        self._episode_reward = 0.0

        self.init_state, _, _, _, self.metadata = self.init_func(self.config['input_height'], self.config['input_width'], self.config['wall_prob'])
        assert self.init_state is not None

        self.karel_world.clear_history()
        self.karel_world.set_new_state(self.init_state, self.metadata)
        self.a_h = []
        self.s_image_h = []
        self._episode_steps = 0
        state_str = self.karel_world.print_state()
        self.initial_obv = self.init_state.copy()

        self.state = self.initial_obv
        
        # Log info about the initial state
        info = {}
        # info = {'r': 0.0, 'a': None, 's': self.state, 'done': False}
        
        return self.initial_obv, info 

    def render(self, mode='video'):
        # raise NotImplementedError('Yet to generate video of predicted program execution')
        if mode == "init_states":
            return self.initial_obv
        elif mode == "video":
            return self.s_image_h
        elif mode == "current_state":
            return self.state
        else:
            raise NotImplementedError('render mode not recognized')
import pddlgym
from pddlgym.structs import Literal
import gym
import numpy as np
from PIL import Image
import gymnasium
import itertools

class RenderObservationWrapper(gym.Wrapper):
    def __init__(self, env, obs):
        super(RenderObservationWrapper, self).__init__(env)
        self.env = env
        self.obs = obs
        self.num_steps = 0
        # You need to define the observation space based on your environment's render output.
        # For instance, if your render returns 800x600 RGB images:
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(480,480,4), dtype=np.uint8)

        num_predicates = self.env.action_space.num_predicates
        objects = self.env.action_space.get_objects()
        num_objects = len(objects)

        self.object_dict = {}
        predicates = self.env.action_space.predicates
        for obj in objects:
            self.object_dict[obj.var_type] = self.object_dict.get(obj.var_type, []) + [obj]
        self.action_list = []
        for act_predi in predicates:
            # For each action, get all possible objects for each argument type
            object_lists = [self.object_dict[var_type] for var_type in act_predi.var_types]

            # Use itertools.product to generate all combinations of objects
            combinations = list(itertools.product(*object_lists))

            # Add these combinations to the action variable list
            for comb in combinations:
                self.action_list.append(Literal(act_predi, comb))

        self.action_space = gymnasium.spaces.Discrete(len(self.action_list))


    def process_render(self):
        # Convert render output to numpy array
        render = self.env.render()
        image = Image.fromarray(render)
        # Optionally, resize or process the image as required
        # image = image.resize((width, height))
        return np.array(image)

    def step(self, action):
        self.num_steps += 1
        # self.env.action_space.all_ground_literals(self.obs)

        # objects = list(self.env.action_space.get_objects())
        # FIXME
        # if len(objects) <= action[1]:
        #     action_object = objects[-1]
        # else:
        #     action_object = objects[action[1]]
        # if len(objects) <= action[2]:
        #     action_object2 = objects[-1]
        # else:
        #     action_object2 = objects[action[2]]
        # if self.env.action_space.predicates[action[0]].arity == 2:
        #     action_literal = self.env.action_space.predicates[action[0]](action_object2, action_object)
        # else:
        #     action_literal = self.env.action_space.predicates[action[0]](action_object)

        action_literal = self.action_list[action]

        obs, reward, done, info = self.env.step(action_literal)
        self.obs = obs
        if done: True
        if self.num_steps >= 100:
            done = True
        return self.process_render(), reward, done, False, info

    def reset(self, **kwargs):
        self.env.reset()
        self.num_steps = 0
        return self.process_render(), []
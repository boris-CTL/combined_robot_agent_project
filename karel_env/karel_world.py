import os
import numpy as np
import scipy
from scipy import spatial
from collections import deque
import random

MAX_NUM_MARKER_topoff = 2
MAX_NUM_MARKER_general = 1

state_table = {
    0: 'Karel facing North',
    1: 'Karel facing East',
    2: 'Karel facing South',
    3: 'Karel facing West',
    4: 'Wall',
    5: '0 marker',
    6: '1 marker',
    7: '2 markers',
}

action_table = {
    0: 'Move',
    1: 'Turn left',
    2: 'Turn right',
    3: 'Pick up a marker',
    4: 'Put a marker'
}


class Karel_world(object):

    def __init__(self, s=None, make_error=True, env_task="program", reward_diff=False, final_reward_scale=True):
        if s is not None:
            self.set_new_state(s)
        self.make_error = make_error
        self.env_task = env_task
        self.rescale_reward = True
        self.final_reward_scale = final_reward_scale
        self.reward_diff = reward_diff
        self.num_actions = len(action_table)
        self.elapse_step = 0
        self.progress_ratio = 0.0

    def set_new_state(self, s, metadata=None):
        self.elapse_step = 0
        self.perception_count = 0
        self.progress_ratio = 0.0
        self.s = s.astype(np.bool_)
        self.s_h = [self.s.copy()]
        self.a_h = []
        self.h = self.s.shape[0]
        self.w = self.s.shape[1]
        p_v = self.get_perception_vector()
        self.p_v_h = [p_v.copy()]
        self.pos_h = [tuple(self.get_location()[:2])]
        self.pos_h_set = set(self.pos_h)
        self.snake_body = deque([(1, 1), (1, 2)])
        self.snake_len  = 2

        if self.env_task != "program":
            self.r_h = []
            self.d_h = []
            self.progress_h = []
            self.program_reward = 0.0
            self.prev_pos_reward = 0.0
            self.init_pos_reward = 0.0
            self.done = False
            # self.stage = 0 # For key2door
            self.metadata = metadata
            self.total_markers = np.sum(s[:,:,6:])
            if self.env_task == 'snake':
                self.snake_marker_pointer = self.metadata['marker_pointer']
                self.snake_marker_list = self.metadata['marker_list']
            #print(self.snake_marker_list)
            #print(self.snake_marker_pointer)
    ###################################
    ###    Collect Demonstrations   ###
    ###################################

    def clear_history(self):
        self.perception_count = 0
        self.elapse_step = 0
        self.progress_ratio = 0.0
        self.s_h = [self.s.copy()]
        self.a_h = []
        self.p_v_h = []
        self.pos_h = [tuple(self.get_location()[:2])]
        self.pos_h_set = set(self.pos_h)
        self.snake_body = deque([(1, 1), (1, 2)])
        self.snake_len  = 2

        if self.env_task != "program":
            self.r_h = []
            self.progress_h = []
            self.d_h = []
            self.program_reward = 0.0
            self.prev_pos_reward = 0.0
            self.init_pos_reward = 0.0
            self.done = False
            self.total_markers = np.sum(self.s_h[-1][:,:,6:])

    def add_to_history(self, a_idx, agent_pos, made_error=False):
        self.s_h.append(self.s.copy())
        self.a_h.append(a_idx)
        p_v = self.get_perception_vector()
        self.p_v_h.append(p_v.copy())

        self.elapse_step += 1
        
        self.total_markers = np.sum(self.s[:,:,6:]) 
        #if self.env_task != 'program' and not made_error:
        #    if a_idx == 3: self.total_markers -= 1
        #    if a_idx == 4: self.total_markers += 1

        if self.env_task != "program":
            reward, done = self._get_state_reward(agent_pos, made_error)
            # log agent position
            pos_tuple = tuple(agent_pos[:2])
            self.pos_h_set.add(pos_tuple)
            self.pos_h.append(pos_tuple)           
            
            self.done = self.done or done
            self.r_h.append(reward)
            self.progress_h.append(self.progress_ratio)
            self.d_h.append(done)
            self.program_reward += reward


    def _get_cleanHouse_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done

        done = False
        pick_marker = 0
        w = self.w
        h = self.h
        state = self.s_h[-1]

        for mpos in self.metadata['marker_positions']:
            if state[mpos[0], mpos[1], 5] and not state[mpos[0], mpos[1], 6]:
                pick_marker += 1

        current_progress_ratio = pick_marker / float(len(self.metadata['marker_positions']))
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = pick_marker == len(self.metadata['marker_positions'])

        reward = reward if self.env_task == 'cleanHouse' else float(done)
        self.done = self.done or done
        return reward, done

    def _get_harvester_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done

        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]

        # calculate total_1 marker in the state
        max_markers = (w-2)*(h-2)

        assert max_markers >= self.total_markers, "max_marksers: {}, self.total_markers: {}".format(max_markers, self.total_markers)
        
        current_progress_ratio = (max_markers - self.total_markers) / float(max_markers)
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = self.total_markers == 0

        reward = reward if self.env_task == 'harvester' else float(done)
        self.done = self.done or done
        return reward, done

    def _get_randomMaze_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done

        done = False
        distance_to_goal = 0
        w = self.w
        h = self.h
        state = self.s_h[-1]

        # initial marker position
        init_state = self.s_h[0]
        x, y = np.where(init_state[:, :, 6] > 0)
        if len(x) != 1: assert 0, '{} markers found!'.format(len(x))
        marker_pos = np.asarray([x[0], y[0]])
        distance_to_goal = -1 * spatial.distance.cityblock(agent_pos[:2], marker_pos)

        done = distance_to_goal == 0
        reward = float(done)
        self.done = self.done or done
        return reward, done

    def _get_fourCorners_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done

        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]
        correct_markers = 0
        reward = 0.0
        
        assert not done and not self.done

        # calculate correct markers
        if state[1, 1, 6]:
            correct_markers += 1
        if state[h-2, 1, 6]:
            correct_markers += 1
        if state[h-2, w-2, 6]:
            correct_markers += 1
        if state[1, w-2, 6]:
            correct_markers += 1
        
        self.total_markers = np.sum(self.s[:,:,6:]) 
        assert self.total_markers >= correct_markers, "total_markers: {}, correct_markers: {}".format(self.total_markers, correct_markers)
        #give zero reward if agent places marker anywhere else
        incorrect_markers = self.total_markers - correct_markers
       
        current_progress_ratio = correct_markers / 4.0
        #if current_progress_ratio > self.progress_ratio:
        #    reward = current_progress_ratio - self.progress_ratio
        #    self.progress_ratio = current_progress_ratio
            
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio


        if incorrect_markers > 0 and reward > 0.0:
            reward = 0.0

        done = correct_markers == 4 or incorrect_markers > 0 
        
        if self.env_task == 'fourCorners_sparse':
            reward = reward if done and not self.done else 0
        self.done = self.done or done
        return reward, done

    def _get_stairClimber_task_reward(self, agent_pos):
        # check if already done
        assert self.reward_diff == True

        if self.done:
            return 0.0, self.done

        done = False
        state = self.s_h[-1]

        # initial marker position
        init_state = self.s_h[0]
        x, y = np.where(init_state[:, :, 6] > 0)
        if len(x) != 1: assert 0, '{} markers found!'.format(len(x))
        marker_pos = np.asarray([x[0], y[0]])
        reward = -1 * spatial.distance.cityblock(agent_pos[:2], marker_pos)
        
        # initial agent position
        x, y, z = np.where(self.s_h[0][:, :, :4] > 0)
        init_pos = np.asarray([x[0], y[0], z[0]])
        longest_distance = spatial.distance.cityblock(init_pos[:2], marker_pos)
        assert longest_distance >= 1.0

        # NOTE: need to do this to avoid high negative reward for first action
        if len(self.s_h) == 2:
            x, y, z = np.where(self.s_h[0][:, :, :4] > 0)
            init_pos = np.asarray([x[0], y[0], z[0]])
            self.prev_pos_reward = -1 * spatial.distance.cityblock(init_pos[:2], marker_pos)

        if not self.reward_diff:
            # since reward is based on manhattan distance, rescale it to range between 0 to 1
            if self.rescale_reward:
                from_min, from_max, to_min, to_max = -(sum(self.s.shape[:2])), 0, -1, 0
                reward = ((reward - from_min) * (to_max - to_min) / (from_max - from_min)) + to_min
            if tuple(agent_pos[:2]) not in self.metadata['agent_valid_positions']:
                reward = -0.2 # -1.0
            done = reward == 0
        else:
            abs_reward = reward
            reward = self.prev_pos_reward-1.0 if tuple(agent_pos[:2]) not in self.metadata['agent_valid_positions'] else reward
            reward = (reward - self.prev_pos_reward) / longest_distance
            assert reward < 1.0, "agent pos: {}, marker_pos: {}, reward: {}, prev_pos_reward: {}".format(agent_pos[:2], marker_pos, reward, self.prev_pos_reward)
            #reward = -1.0 if tuple(agent_pos[:2]) not in self.metadata['agent_valid_positions'] else reward
            self.prev_pos_reward = abs_reward
            done = abs_reward == 0

        # # calculate previous distance to the goal
        # # TODO: check why this work
        # x, y, z = np.where(self.s_h[-2][:, :, :4] > 0)
        # prev_pos = np.asarray([x[0], y[0], z[0]])
        # self.prev_pos_reward = -1 * spatial.distance.cityblock(prev_pos[:2], marker_pos)


        # current_progress_ratio = (distance_to_goal - self.prev_pos_reward) / (self.init_pos_reward * -1)
        # reward = current_progress_ratio - self.progress_ratio
        # self.progress_ratio = current_progress_ratio
        # reward = -1.0 if tuple(agent_pos[:2]) not in self.metadata['agent_valid_positions'] else reward
        # self.prev_pos_reward = distance_to_goal
        # done = distance_to_goal == 0

        reward = reward if self.env_task == 'stairClimber' else float(done)
        if self.env_task == 'stairClimber_sparse':
            reward = reward if done and not self.done else 0
        self.done = self.done or done
        return reward, done

    def _get_topOff_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done       

        # assert self.reward_diff
        done = False
        score = 0
        w = self.w
        h = self.h
        state = self.s_h[-1]

        for c in range(1, agent_pos[1]+1):
            if (h-2, c) in self.metadata['not_expected_marker_positions']:
                if state[h-2, c, 7]:
                    score += 1
                else:
                    break
            else:
                assert (h-2, c) in self.metadata['expected_marker_positions']
                if state[h-2, c, 5]:
                    score += 1
                else:
                    break

        if (self.w - 2 == agent_pos[1] and self.h - 2 == agent_pos[0]) and score == w-2:
            score += 1

        current_progress_ratio = score / (w-1)
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = sum([state[pos[0], pos[1], 7] for pos in self.metadata['not_expected_marker_positions']]) == len(
            self.metadata['not_expected_marker_positions']) and (self.w - 2 == agent_pos[1] and self.h - 2 == agent_pos[0]) and current_progress_ratio==1.0

        reward = reward if self.env_task == 'topOff' else float(done)
        if self.env_task == 'topOff_sparse':
            reward = reward if done and not self.done else 0
        self.done = self.done or done
        return reward, done


    def _get_randomMaze_key2door_task_reward(self, agent_pos): ## key2door
        # check if already done
        if self.done:
            return 0.0, self.done

        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]

        '''
        # initial marker position
        init_state = self.s_h[0]
        x, y = np.where(init_state[:, :, 6] > 0)
        if len(x) != 2: assert 0, '{} markers found!'.format(len(x))

        mxs, mys = np.where(state[:, :, 6] > 0)
        if len(mxs):
            for mx, my in zip(mxs, mys):
                if (mx, my) != (x[0], y[0]) and (mx, my) != (x[1], y[1]):
                    self.done = 1
                    if self.stage == 0:
                        return -0.1, self.done
                    else:
                        return -0.1, self.done

        prev_stage = self.stage
        if self.stage == 0:
            if state[x[0], y[0], 7] or state[x[1], y[1], 7]:
                self.done = 1
                return -0.1, self.done
            elif len(mxs) < 2:
                self._door = (mxs[0], mys[0])
                self._key = (x[0], y[0]) if (self._door == (x[1], y[1])) else (x[1], y[1])
                self.stage = 1
        elif self.stage == 1:
            if not state[self._key[0], self._key[1], 5]:
                self.done = 1
                return -0.1, self.done
            if state[self._door[0], self._door[1], 5]:
                self.done = 1
                return -0.1, self.done
            if state[self._door[0], self._door[1], 7]:
                self.stage = 2
                done = True
        '''
        total_markers = np.sum(state[:,:,6:])
        error_markers = total_markers - 2
        score = 0
        if state[6, 3, 5]: # [1, 3, 7]
            score += 0.5
        if state[6, 3, 5] and state[1, 6, 7]:
            score += 0.5
        
        #for y in range(1, 6):
        #    if state[1, y, 6] or state[1, y, 7]:
        #        score -= 0.1
        #for x in range(2, 6):
        #    if state[x, 3, 6] or state[x, 3, 7]:
        #        score -= 0.1
        if error_markers > 0:
            score -= error_markers * 0.0001

        #if state[6, 3, 7]:
        #    score -= 0.1
        #if state[1, 6, 5]:
        #    score -= 0.1

        current_progress_ratio = score
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = (current_progress_ratio==1.0)

        reward = reward if self.env_task == 'randomMaze_key2door' else float(done)
        if self.env_task == 'randomMaze_key2door_sparse':
            reward = reward if done and not self.done else 0
                
        self.done = self.done or done
        return reward, done

   
    def _get_randomMaze_key2doorSpace_task_reward(self, agent_pos): 
        # check if already done
        if self.done:
            return 0.0, self.done

        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]

        total_markers = np.sum(state[:,:,6:])
        error_markers = total_markers - 2
        score = 0
        if state[6, 3, 5]: # [1, 3, 7]
            score += 0.5
        if state[6, 3, 5] and state[1, 6, 7]:
            score += 0.5
        
        #for y in range(1, 6):
        #    if state[1, y, 6] or state[1, y, 7]:
        #        score -= 0.1
        #for x in range(2, 6):
        #    if state[x, 3, 6] or state[x, 3, 7]:
        #        score -= 0.1
        if error_markers > 0:
            score -= error_markers * 0.0001 # penalty

        #if state[6, 3, 7]:
        #    score -= 0.1
        #if state[1, 6, 5]:
        #    score -= 0.1

        current_progress_ratio = score
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = (current_progress_ratio==1.0)

        reward = reward if self.env_task == 'randomMaze_key2doorSpace' else float(done)
        if self.env_task == 'randomMaze_key2doorSpace_sparse':
            reward = reward if done and not self.done else 0
                
        self.done = self.done or done
        return reward, done


    def _get_oneStroke_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done
 
        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]
        pos_tuple = tuple(agent_pos[:2])
        
        is_overlap  = pos_tuple in self.pos_h_set and pos_tuple != self.pos_h[-1]
        is_hit_wall = self.a_h[-1] == 0 and pos_tuple == self.pos_h[-1]
        traverse_length = len(self.pos_h_set)

        # calculate total_1 marker in the state
        max_markers = (w-2)*(h-2)

        assert max_markers >= self.total_markers, "max_marksers: {}, self.total_markers: {}".format(max_markers, self.total_markers)
        assert len(self.pos_h) >= len(self.pos_h_set), "self.pos_h:{}, self.pos_h_set: {}".format(self.pos_h, self.pos_h_set)

        current_progress_ratio = traverse_length / float(max_markers)
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = is_overlap or is_hit_wall or traverse_length == max_markers

        reward = reward if self.env_task == 'oneStroke' else float(done)
        self.done = self.done or done
        return reward, done

    def _get_doorkey_task_reward(self, agent_pos): ## doorkey
        # check if already done
        if self.done:
            return 0.0, self.done

        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]

        total_markers = np.sum(state[:,:,6:])
        error_markers = total_markers - 2
        score = 0
        if state[self.metadata['key'][0], self.metadata['key'][1], 5]: # [5, 2, 5]  # [1, 3, 7]
            score += 0.5
        if state[self.metadata['key'][0], self.metadata['key'][1], 5] and state[self.metadata['target'][0], self.metadata['target'][1], 7]:
            score += 0.5

        # open the door if marker picked
        if state[self.metadata['key'][0], self.metadata['key'][1], 5]:
            for door_pos in self.metadata['door_positions']: 
                self.s[door_pos[0], door_pos[1], 4] = False

        if error_markers > 0:
            score -= error_markers * 0.0001 # penalty

        current_progress_ratio = score
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = (current_progress_ratio==1.0) #or error_markers > 0 

        reward = reward if self.env_task == 'doorkey' else float(done)
        if self.env_task == 'doorkey_sparse':
            reward = reward if done and not self.done else 0
                
        self.done = self.done or done
        return reward, done

    def _get_seeder_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done

        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]

        # calculate total_1 marker in the state
        existing_marker_num = len(self.metadata['existing_marker'])
        
        max_markers = (w-2)*(h-2) - existing_marker_num
       
        total_one_markers = np.sum(self.s[:,:,6])
        total_two_markers = np.sum(self.s[:,:,7])
        
        score = total_one_markers - existing_marker_num # - total_two_markers * 3

        current_progress_ratio = score / float(max_markers)
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = (total_one_markers == max_markers and total_two_markers == 0) or total_two_markers > 0

        reward = reward if self.env_task == 'seeder' else float(done)
        self.done = self.done or done
        return reward, done

    def _get_snake_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done
 
        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]
        pos_tuple = tuple(agent_pos[:2])
        
        #is_hit_wall = self.a_h[-1] == 0 and pos_tuple == self.pos_h[-1]
        is_hit_body = self.s[agent_pos[0], agent_pos[1], 7]

        assert self.snake_len >= len(self.snake_body), "self.snake_len:{}, self.snake_body: {}".format(self.snake_len, self.snake_body)
        
        current_progress_ratio = (self.snake_len - 2) / 20.0 # max marker eatable: 10 (max snake length: 22)
        #current_progress_ratio = (2.0 ** ((self.snake_len - 2) / 4.0) -1) / 31.0 # max marker eatable: 5 (max snake length: 2 + 4*5 = 22)
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = is_hit_body or current_progress_ratio >= 0.99 #is_hit_wall or current_progress_ratio >= 0.99

        reward = reward if self.env_task == 'snake' else float(done)
        self.done = self.done or done
        return reward, done


    def _get_state_reward(self, agent_pos, made_error=False):
        if self.env_task == 'cleanHouse' or self.env_task == 'cleanHouse_sparse':
            reward, done = self._get_cleanHouse_task_reward(agent_pos)
        elif self.env_task == 'harvester' or self.env_task == 'harvester_sparse':
            reward, done = self._get_harvester_task_reward(agent_pos)
        elif self.env_task == 'fourCorners' or self.env_task == 'fourCorners_sparse':
            reward, done = self._get_fourCorners_task_reward(agent_pos)
        elif self.env_task == 'randomMaze' or self.env_task == 'randomMaze_sparse':
            reward, done = self._get_randomMaze_task_reward(agent_pos)
        elif self.env_task == 'stairClimber' or self.env_task == 'stairClimber_sparse':
            reward, done = self._get_stairClimber_task_reward(agent_pos)
        elif self.env_task == 'topOff' or self.env_task == 'topOff_sparse':
            reward, done = self._get_topOff_task_reward(agent_pos)
        elif self.env_task == 'randomMaze_key2door' or self.env_task == 'randomMaze_key2door_sparse': 
            reward, done = self._get_randomMaze_key2door_task_reward(agent_pos)
        elif self.env_task == 'randomMaze_key2doorSpace' or self.env_task == 'randomMaze_key2doorSpace_sparse': 
            reward, done = self._get_randomMaze_key2doorSpace_task_reward(agent_pos)
        elif self.env_task == 'oneStroke' or self.env_task == 'oneStroke_sparse': ## oneStroke
            reward, done = self._get_oneStroke_task_reward(agent_pos)
        elif self.env_task == 'doorkey' or self.env_task == 'doorkey_sparse': ## doorkey
            reward, done = self._get_doorkey_task_reward(agent_pos)
        elif self.env_task == 'seeder' or self.env_task == 'seeder_sparse':
            reward, done = self._get_seeder_task_reward(agent_pos)
        elif self.env_task == 'snake' or self.env_task == 'snake_sparse':
            reward, done = self._get_snake_task_reward(agent_pos)
        else:
            raise NotImplementedError('{} task not yet supported'.format(self.env_task))

        return reward, done

    def print_state(self, state=None):
        agent_direction = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
        state = self.s_h[-1] if state is None else state
        state_2d = np.chararray(state.shape[:2])
        state_2d[:] = '.'
        state_2d[state[:,:,4]] = 'x'
        state_2d[state[:,:,6]] = 'm'
        state_2d[state[:,:,7]] = 'M'
        x, y, z = np.where(state[:, :, :4] > 0)
        state_2d[x[0], y[0]] = agent_direction[z[0]]

        state_2d = state_2d.decode()
        state_str = ""
        for i in range(state_2d.shape[0]):
            state_str += "".join(state_2d[i]) 
            state_str += "\n"
        # print(state_str)
        return state_str

    def render(self, mode='rgb_array'):
        return self.s_h[-1]

    # get location (x, y) and facing {north, east, south, west}
    def get_location(self):
        x, y, z = np.where(self.s[:, :, :4] > 0)
        return np.asarray([x[0], y[0], z[0]])

    # get the neighbor {front, left, right} location
    def get_neighbor(self, face):
        loc = self.get_location()
        if face == 'front':
            neighbor_loc = loc[:2] + {
                0: [-1, 0],
                1: [0, 1],
                2: [1, 0],
                3: [0, -1]
            }[loc[2]]
        elif face == 'left':
            neighbor_loc = loc[:2] + {
                0: [0, -1],
                1: [-1, 0],
                2: [0, 1],
                3: [1, 0]
            }[loc[2]]
        elif face == 'right':
            neighbor_loc = loc[:2] + {
                0: [0, 1],
                1: [1, 0],
                2: [0, -1],
                3: [-1, 0]
            }[loc[2]]
        return neighbor_loc

    ###################################
    ###    Perception Primitives    ###
    ###################################
    # return if the neighbor {front, left, right} of Karel is clear
    def neighbor_is_clear(self, face):
        self.perception_count += 1
        neighbor_loc = self.get_neighbor(face)
        if neighbor_loc[0] >= self.h or neighbor_loc[0] < 0 \
                or neighbor_loc[1] >= self.w or neighbor_loc[1] < 0:
            return False
        return not self.s[neighbor_loc[0], neighbor_loc[1], 4]

    def front_is_clear(self):
        return self.neighbor_is_clear('front')

    def left_is_clear(self):
        return self.neighbor_is_clear('left')

    def right_is_clear(self):
        return self.neighbor_is_clear('right')

    # return if there is a marker presented
    def marker_present(self):
        self.perception_count += 1
        loc = self.get_location()
        return np.sum(self.s[loc[0], loc[1], 6:]) > 0

    def no_marker_present(self):
        self.perception_count += 1
        loc = self.get_location()
        return np.sum(self.s[loc[0], loc[1], 6:]) == 0

    def get_perception_list(self):
        vec = ['frontIsClear', 'leftIsClear',
               'rightIsClear', 'markersPresent',
               'noMarkersPresent']
        return vec

    def get_perception_vector(self):
        vec = [self.front_is_clear(), self.left_is_clear(),
               self.right_is_clear(), self.marker_present(),
               self.no_marker_present()]
        return np.array(vec)

    ###################################
    ###       State Transition      ###
    ###################################
    # given a state and a action, return the next state
    def state_transition(self, a_idx):
        made_error = False
        # a_idx = np.argmax(a)
        loc = self.get_location()

        if a_idx == 0:
            # move
            if self.front_is_clear():
                front_loc = self.get_neighbor('front')
                loc_vec = self.s[loc[0], loc[1], :4]
                self.s[front_loc[0], front_loc[1], :4] = loc_vec
                self.s[loc[0], loc[1], :4] = np.zeros(4) > 0
                assert np.sum(self.s[front_loc[0], front_loc[1], :4]) > 0
 
                if self.env_task == "oneStroke" or self.env_task == "snake" and not self.done:
                    self.s[loc[0], loc[1], 7]  = True # change passed grid to double marker
                    self.s[loc[0], loc[1], 6]  = False
                    self.s[loc[0], loc[1], 5]  = False
                    
                    if self.env_task == "oneStroke":
                        self.s[loc[0], loc[1], 7]  = False
                        self.s[loc[0], loc[1], 4]  = True # change passed grid to wall
  
                if self.env_task == "snake" and not self.done:
                    if (front_loc[0], front_loc[1]) not in self.snake_body:
                        self.snake_body.append((loc[0], loc[1]))
                    else:
                        self.done = True
                        return
                    assert len(self.snake_body) >= 2
                    if self.s[front_loc[0], front_loc[1], 6] and self.snake_len < 22: # max snake length = 22
                        self.snake_len += 1 # += 2
                        self.s[front_loc[0], front_loc[1], 6] = False
                        # generate new marker
                        dummy_check = 0
                        while True:
                            #m_pos = np.random.randint(1, 7, size=[2])
                            #if (m_pos[0], m_pos[1]) != (loc[0], loc[1]) and np.sum(self.s[m_pos[0], m_pos[1], :5]) <= 0:
                            #    self.s[m_pos[0], m_pos[1], 6] = True
                            #    break
                            m_pos = self.snake_marker_list[self.snake_marker_pointer]
                            self.snake_marker_pointer = (self.snake_marker_pointer + 1) % len(self.snake_marker_list)
                            if np.sum(self.s[m_pos[0], m_pos[1], : ]) <= 0:
                                self.s[m_pos[0], m_pos[1], 6] = True
                                break
                            dummy_check += 1
                            if dummy_check > 50:
                                print("snake length: ", self.snake_len)
                                self.print_state()
                                assert False, "no valid marker found, m_pos: {}, state at m_pos:{}".format(m_pos, self.s[m_pos[0], m_pos[1]])
 
                    # check if snake tail should disappear
                    if len(self.snake_body) > self.snake_len:  
                        tail_pos = self.snake_body.popleft()
                        self.s[tail_pos[0], tail_pos[1], :] = np.zeros(len(state_table)) > 0

                assert np.sum(self.s[front_loc[0], front_loc[1], :4]) > 0
                next_loc = front_loc
            else:
                if self.make_error:
                    raise RuntimeError("Failed to move.")
                loc_vec = np.zeros(4) > 0
                loc_vec[(loc[2] + 2) % 4] = True  # Turn 180
                self.s[loc[0], loc[1], :4] = loc_vec
                next_loc = loc
            self.add_to_history(a_idx, next_loc)
        elif a_idx == 1 or a_idx == 2:
            # turn left or right
            loc_vec = np.zeros(4) > 0
            loc_vec[(a_idx * 2 - 3 + loc[2]) % 4] = True
            self.s[loc[0], loc[1], :4] = loc_vec
            self.add_to_history(a_idx, loc)

        elif a_idx == 3 or a_idx == 4:
            # pick up or put a marker
            num_marker = np.argmax(self.s[loc[0], loc[1], 5:])
            # just clip the num of markers for now
            if self.env_task in [
                    'topOff', 'topOff_sparse', 
                    'randomMaze_key2door', 'randomMaze_key2door_sparse', 
                    'randomMaze_key2doorSpace', 'randomMaze_key2doorSpace_sparse', 
                    'doorkey', 'doorkey_sparse', 
                    'seeder', 'seeder_sparse'
                    ]:
                new_num_marker = np.clip(a_idx*2-7 + num_marker, 0, MAX_NUM_MARKER_topoff)
            else:
                new_num_marker = np.clip(a_idx*2-7 + num_marker, 0, MAX_NUM_MARKER_general)
            #new_num_marker = a_idx*2-7 + num_marker
            #if new_num_marker < 0:
            #    if self.make_error:
            #        raise RuntimeError("No marker to pick up.")
            #    else:
            #        new_num_marker = num_marker
            #    made_error = True
            #elif new_num_marker > MAX_NUM_MARKER-1:
            #    if self.make_error:
            #        raise RuntimeError("Cannot put more marker.")
            #    else:
            #        new_num_marker = num_marker
            #    made_error = True
            marker_vec = np.zeros(MAX_NUM_MARKER_topoff+1) > 0
            marker_vec[new_num_marker] = True
            self.s[loc[0], loc[1], 5:] = marker_vec
            self.add_to_history(a_idx, loc, made_error)
        else:
            raise RuntimeError("Invalid action")
        return

    # given a karel env state, return a visulized image
    def state2image(self, s=None, grid_size=100, root_dir='./'):
        h = s.shape[0]
        w = s.shape[1]
        img = np.ones((h*grid_size, w*grid_size, 1))
        import pickle
        from PIL import Image
        import os.path as osp
        f = pickle.load(open(osp.join(root_dir, 'karel_env/asset/texture.pkl'), 'rb'))
        wall_img = f['wall'].astype('uint8')
        marker_img = f['marker'].astype('uint8')
        agent_0_img = f['agent_0'].astype('uint8')
        agent_1_img = f['agent_1'].astype('uint8')
        agent_2_img = f['agent_2'].astype('uint8')
        agent_3_img = f['agent_3'].astype('uint8')
        blank_img = f['blank'].astype('uint8')
        #blanks
        for y in range(h):
            for x in range(w):
                img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = blank_img
        # wall
        y, x = np.where(s[:, :, 4])
        for i in range(len(x)):
            img[y[i]*grid_size:(y[i]+1)*grid_size, x[i]*grid_size:(x[i]+1)*grid_size] = wall_img
        # marker
        y, x = np.where(np.sum(s[:, :, 6:], axis=-1))
        for i in range(len(x)):
            img[y[i]*grid_size:(y[i]+1)*grid_size, x[i]*grid_size:(x[i]+1)*grid_size] = marker_img
        # karel
        y, x = np.where(np.sum(s[:, :, :4], axis=-1))
        if len(y) == 1:
            y = y[0]
            x = x[0]
            idx = np.argmax(s[y, x])
            marker_present = np.sum(s[y, x, 6:]) > 0
            if marker_present:
                extra_marker_img = Image.fromarray(f['marker'].squeeze()).copy()
                if idx == 0:
                    extra_marker_img.paste(Image.fromarray(f['agent_0'].squeeze()))
                    extra_marker_img = f['marker'].squeeze() + f['agent_0'].squeeze()
                    extra_marker_img = np.minimum(f['marker'].squeeze() , f['agent_0'].squeeze())
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = np.expand_dims(np.array(extra_marker_img), axis=-1)
                elif idx == 1:
                    extra_marker_img.paste(Image.fromarray(f['agent_1'].squeeze()))
                    extra_marker_img = f['marker'].squeeze() + f['agent_1'].squeeze()
                    extra_marker_img = np.minimum(f['marker'].squeeze() , f['agent_1'].squeeze())
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = np.expand_dims(np.array(extra_marker_img), axis=-1)
                elif idx == 2:
                    extra_marker_img.paste(Image.fromarray(f['agent_2'].squeeze()))
                    extra_marker_img = f['marker'].squeeze() + f['agent_2'].squeeze()
                    extra_marker_img = np.minimum(f['marker'].squeeze() , f['agent_2'].squeeze())
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = np.expand_dims(np.array(extra_marker_img), axis=-1)
                elif idx == 3:
                    extra_marker_img.paste(Image.fromarray(f['agent_3'].squeeze()))
                    extra_marker_img = f['marker'].squeeze() + f['agent_3'].squeeze()
                    extra_marker_img = np.minimum(f['marker'].squeeze() , f['agent_3'].squeeze())
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = np.expand_dims(np.array(extra_marker_img), axis=-1)
            else:
                if idx == 0:
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['agent_0']
                elif idx == 1:
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['agent_1']
                elif idx == 2:
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['agent_2']
                elif idx == 3:
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['agent_3']
        elif len(y) > 1:
            raise ValueError
        return img

    def state2str(self, s=None, grid_size=100, root_dir='./'):
        '''
        Given a state, return a text description of the state
        '''
        h = s.shape[0]
        w = s.shape[1]

        state_str = ""
        # describing the state
        for y in range(h):
            for x in range(w):
                if s[y, x, 4]:
                    state_str += "x"
                elif s[y, x, 6]:
                    state_str += "m"
                elif s[y, x, 7]:
                    state_str += "M"
                elif s[y, x, 0]:
                    state_str += "N"
                elif s[y, x, 1]:
                    state_str += "E"
                elif s[y, x, 2]:
                    state_str += "S"
                elif s[y, x, 3]:
                    state_str += "W"
                else:
                    state_str += "."
            state_str += "\n"

        # karel
        y, x = np.where(np.sum(s[:, :, :4], axis=-1))
        if len(y) > 1:
            raise ValueError

        return state_str
        
        



class KarelStateGenerator(object):
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def print_state(self, state=None):
        agent_direction = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
        state_2d = np.chararray(state.shape[:2])
        state_2d[:] = '.'
        state_2d[state[:,:,4]] = 'x'
        state_2d[state[:,:,6]] = 'M'
        x, y, z = np.where(state[:, :, :4] > 0)
        state_2d[x[0], y[0]] = agent_direction[z[0]]

        state_2d = state_2d.decode()
        for i in range(state_2d.shape[0]):
            print("".join(state_2d[i]))

    # generate an initial env
    def generate_single_state(self, h=8, w=8, wall_prob=0.2, env_task_metadata={}):
        s = np.zeros([h, w, len(state_table)]) > 0
        # Wall
        s[:, :, 4] = self.rng.rand(h, w) > 1 - wall_prob
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True
        # Karel initial location
        valid_loc = False
        while(not valid_loc):
            y = self.rng.randint(0, h)
            x = self.rng.randint(0, w)
            if not s[y, x, 4]:
                valid_loc = True
                s[y, x, self.rng.randint(0, 4)] = True
        # Marker: num of max marker == 2 for now
        s[:, :, 6] = (self.rng.rand(h, w) > 0.85) * (s[:, :, 4] == False) > 0
        s[:, :, 5] = 1 - (np.sum(s[:, :, 6:], axis=-1) > 0) > 0
        assert np.sum(s[:, :, 5:]) == h*w, np.sum(s[:, :, :5])
        marker_weight = np.reshape(np.array(range(3)), (1, 1, 3))
        return s, y, x, np.sum(s[:, :, 4]), np.sum(marker_weight*s[:, :, 5:])

    # generate an initial env
    def generate_program_instruct_state(self, h=8, w=8, wall_prob=0.1, env_task_metadata={}, idx=0):
        s = np.zeros([h, w, len(state_table)]) > 0
        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True
        assert idx < (h-2) * (w-2), "program instruct state max number reached, idx: {}, maximum # state: {}".format(idx, (h-2)*(w-2))
        # initial karel position: karel facing east at the last row in environment
        #agent_pos = (h-2, 2)
        for i in range(idx):
            idx_mod = i % (h-2)
            idx_div = i // (h-2)
            agent_pos = ((0+idx_mod)%(h-2)+1, (0+idx_div)%(w-2)+1)
            s[agent_pos[0], agent_pos[1], 1] = True
            s[agent_pos[0], agent_pos[1], 6] = True


        idx_mod = idx % (h-2)
        idx_div = idx // (h-2)
        agent_pos = ((0+idx_mod)%(h-2)+1, (0+idx_div)%(w-2)+1)

        metadata = {}         
        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), np.sum(s[:, :, 5:])


    # generate an initial env for cleanHouse problem
    def generate_single_state_cleanHouse(self, h=14, w=22, wall_prob=0.1, env_task_metadata={}, is_top_off=False):
        """
        initial state generator for cleanHouse problem
        Valid program for cleanHouse problem:
        DEF run m( WHILE c( frontIsClear c) w( IF c( markersPresent c) i( pickMarker i) IFELSE c( leftIsClear c) i( turnLeft i) ELSE e( move e) w) m)

        :param h:
        :param w:
        :param wall_prob:
        :return:
        """

        world_map = [
            ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            ['-',   0,   0, '-', '-', '-', '-', '-', '-', '-', '-', '-', '-',   0, '-',   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0, '-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-', '-', '-', '-',   0, '-', '-'],
            ['-', '-',   0, '-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-'],
            ['-', '-',   0,   0,   0,   0,   0,   0,   0,   0, '-',   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-'],
            ['-', '-', '-',   0, '-',   0, '-', '-', '-',   0, '-',   0,   0, '-', '-', '-',   0, '-',   0, '-', '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0,   0, '-',   0,   0,   0, '-',   0,   0, '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0, '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-', '-',   0, '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-',   0,   0, '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-',   0,   0, '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-',   0,   0, '-', '-'],
            ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
        ]

        assert h == 14 and w == 22, 'karel cleanHouse environment should be 14 x 22, found {} x {}'.format(h, w)
        s = np.zeros([h, w, len(state_table)]) > 0
        # Wall
        s[0, :, 4] = True
        s[h - 1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w - 1, 4] = True

        # Karel initial location
        agent_pos = (1, 13)
        hardcoded_invalid_marker_locations = [(1, 13), (2, 12), (3, 10), (4, 11), (5, 11), (6, 10)]
        s[agent_pos[0], agent_pos[1], 2] = True


        for y1 in range(h):
            for x1 in range(w):
                s[y1, x1, 4] = world_map[y1][x1] == '-'
                s[y1, x1, 5] = True if world_map[y1][x1] != 'M' else False
                s[y1, x1, 6] = True if world_map[y1][x1] == 'M' else False

        expected_marker_positions = set()
        for y1 in range(h):
            for x1 in range(13):
                if s[y1, x1, 4]:
                    if y1 - 1 > 0 and not s[y1 -1, x1, 4]: expected_marker_positions.add((y1 - 1,x1))
                    if y1 + 1 < h - 1 and not s[y1 +1, x1, 4]: expected_marker_positions.add((y1 + 1,x1))
                    if x1 - 1 > 0 and not s[y1, x1 - 1, 4]: expected_marker_positions.add((y1,x1 - 1))
                    if x1 + 1 < 13 - 1 and not s[y1, x1 + 1, 4]: expected_marker_positions.add((y1,x1 + 1))

        # put 2 markers near start point for end condition
        s[agent_pos[0]+1, agent_pos[1]-1, 5] = False
        s[agent_pos[0]+1, agent_pos[1]-1, 7] = True

        # place 10 Markers
        expected_marker_positions = list(expected_marker_positions)
        random.shuffle(expected_marker_positions)
        assert len(expected_marker_positions) >= 10
        marker_positions = []
        for i, mpos in enumerate(expected_marker_positions):
            if mpos in hardcoded_invalid_marker_locations:
                continue
            s[mpos[0], mpos[1], 5] = False
            s[mpos[0], mpos[1], 6] = True
            marker_positions.append(mpos)
            if len(marker_positions) == 10:
                break

        metadata = {'agent_valid_positions': None, 'expected_marker_positions': expected_marker_positions, 'marker_positions': marker_positions}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata


    def generate_single_state_harvester(self, h=8, w=8, wall_prob=0.1, env_task_metadata={}, is_top_off=False):
        """
        initial state generator for harvester problem
        Valid program for harvester problem:
        DEF run m( WHILE c( markersPresent c) w( WHILE c( markersPresent c) w( pickMarker move w) turnRight move turnLeft WHILE c( markersPresent c) w( pickMarker move w) turnLeft move turnRight w) m)

        :param h:
        :param w:
        :param wall_prob:
        :return:
        """
        mode = env_task_metadata.get("mode", "train")
        marker_prob = env_task_metadata.get("train_marker_prob", 1.0) if mode == 'train' else env_task_metadata.get("test_marker_prob", 1.0)


        s = np.zeros([h, w, len(state_table)]) > 0
        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True

        # initial karel position: karel facing east at the last row in environment
        agent_pos = (h-2, 1)
        s[agent_pos[0], agent_pos[1], 1] = True

        # put 1 marker at every location in grid
        if marker_prob == 1.0:
            s[1:h-1, 1:w-1, 6] = True
        else:
            valid_marker_pos = np.array([(r,c) for r in range(1,h-1) for c in range(1,w-1)])
            marker_pos = valid_marker_pos[np.random.choice(len(valid_marker_pos), size=int(marker_prob*len(valid_marker_pos)), replace=False)]
            for pos in marker_pos:
                s[pos[0], pos[1], 6] = True

        metadata = {}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata



    # generate an initial env for randomMaze problem
    def generate_single_state_randomMaze(self, h=8, w=8, wall_prob=0.1, env_task_metadata={}, is_top_off=False):
        """
        initial state generator for random maze problem
        Valid program for random maze problem:
        DEF run m( WHILE c( noMarkersPresent c) w( IFELSE c( rightIsClear c) i( turnRight i) ELSE e( WHILE c( not c( frontIsClear c) c) w( turnLeft w) e) move w) m)

        :param h:
        :param w:
        :param wall_prob:
        :return:
        """

        def get_neighbors(cur_pos, h, w):
            neighbor_list = []
            #neighbor top
            if cur_pos[0] - 2 > 0: neighbor_list.append([cur_pos[0] - 2, cur_pos[1]])
            # neighbor bottom
            if cur_pos[0] + 2 < h - 1: neighbor_list.append([cur_pos[0] + 2, cur_pos[1]])
            # neighbor left
            if cur_pos[1] - 2 > 0: neighbor_list.append([cur_pos[0], cur_pos[1] - 2])
            # neighbor right
            if cur_pos[1] + 2 < w - 1: neighbor_list.append([cur_pos[0], cur_pos[1] + 2])
            return neighbor_list

        s = np.zeros([h, w, len(state_table)]) > 0
        # convert every location to wall
        s[:, :, 4] = True
        #start from bottom left corner
        init_pos = [h - 2, 1]
        visited = np.zeros([h,w])
        stack = []

        # convert initial location to empty location from wall
        # iterative implementation of random maze generator at https://en.wikipedia.org/wiki/Maze_generation_algorithm
        s[init_pos[0], init_pos[1], 4] = False
        visited[init_pos[0], init_pos[1]] = True
        stack.append(init_pos)

        while len(stack) > 0:
            cur_pos = stack.pop()
            neighbor_list = get_neighbors(cur_pos, h, w)
            random.shuffle(neighbor_list)
            for neighbor in neighbor_list:
                if not visited[neighbor[0], neighbor[1]]:
                    stack.append(cur_pos)
                    s[(cur_pos[0]+neighbor[0])//2, (cur_pos[1]+neighbor[1])//2, 4] = False
                    s[neighbor[0], neighbor[1], 4] = False
                    visited[neighbor[0], neighbor[1]] = True
                    stack.append(neighbor)

        # init karel agent position
        agent_pos = init_pos
        s[agent_pos[0], agent_pos[1], 1] = True

        # Marker location
        valid_loc = False
        while (not valid_loc):
           ym = self.rng.randint(0, h)
           xm = self.rng.randint(0, w)
           if xm == 1 and ym == h-2: # added 
               continue
           if not s[ym, xm, 4]:
               valid_loc = True
               s[ym, xm, 6] = True
               assert not s[ym, xm, 4]

        assert not s[ym, xm, 4]

        # put 0 markers everywhere but 1 location
        s[:, :, 5] = 1 - (np.sum(s[:, :, 6:], axis=-1) > 0) > 0
        assert np.sum(s[:, :, 6]) == 1
        metadata = {'agent_valid_positions': None}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

    # generate an initial env for fourCorners problem
    def generate_single_state_fourCorners(self, h=8, w=8, wall_prob=0.1, env_task_metadata={}, is_top_off=False):
        """
        initial state generator for four corners problem
        Valid program for four corners problem:
        DEF run m( WHILE c( noMarkersPresent c) w( WHILE c( frontIsClear c) w( move w) IF c( noMarkersPresent c) i( putMarker turnLeft move i) w) m)

        :param h:
        :param w:
        :param wall_prob:
        :return:
        """
        s = np.zeros([h, w, len(state_table)]) > 0
        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True
        
        
        #agent_direction = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
        init_pos = [(h-3, w-2), (h-2, 2), (2, 1), (1, w-3)]
        init_dir = [[0, 1, 2, 3], [2, 3, 0, 1]]
        init_idx = random.randint(0, 3)
        init_dir_idx = random.randint(0, 1)
        agent_pos = init_pos[init_idx]
        s[agent_pos[0], agent_pos[1], init_dir[init_dir_idx][init_idx]] = True

        metadata = {}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

    # generate an initial env
    def generate_single_state_topOff(self, h=8, w=8, wall_prob=0.1, env_task_metadata={}, is_top_off=True):
        """
        initial state generator for chain smoker and top off problem both
        Valid program for chain smoker problem:
        DEF run m( WHILE c( frontIsClear c) w( IF c( noMarkersPresent c) i( putMarker i) move w) m)
        Valid program for top off problem:
        DEF run m( WHILE c( frontIsClear c) w( IF c( MarkersPresent c) i( putMarker i) move w) m)

        :param h:
        :param w:
        :param wall_prob:
        :return:
        """
        s = np.zeros([h, w, len(state_table)]) > 0
        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True

        # initial karel position: karel facing east at the last row in environment
        agent_pos = (h-2, 1)
        s[agent_pos[0], agent_pos[1], 1] = True

        # randomly put markers at row h-2
        s[h-2, 1:w-1, 6] = self.rng.rand(w-2) > 1 - wall_prob
        # NOTE: need marker in last position as the condition is to check till I reach end
        s[h-2, w-2, 6] = True if not is_top_off else False
        s[:, :, 5] = 1 - (np.sum(s[:, :, 6:], axis=-1) > 0) > 0
        assert np.sum(s[:,:,5:]) == w*h

        # randomly generate wall at h-3 row
        mode = env_task_metadata.get('mode', 'train')
        hash_info_path = env_task_metadata.get('hash_info', None)
        if is_top_off and hash_info_path is not None:
            train_configs = env_task_metadata.get('train_configs', 1.0)
            test_configs = env_task_metadata.get('test_configs', 1.0)
            hash_info = pickle.load(open(hash_info_path,"rb"))
            assert hash_info['w'] == w and hash_info['h'] == h
            hashtable = hash_info['table']
            split_idx = int(len(hashtable)*train_configs) if mode == 'train' else int(len(hashtable)*test_configs)
            hashtable = hashtable[:split_idx] if mode == 'train' else hashtable[-split_idx:]
            key = s[h-2, 1:w-2, 6].tostring()
            if key not in hashtable:
                return self.generate_single_state_chain_smoker(h, w, wall_prob, env_task_metadata, is_top_off)

        # generate valid agent positions
        valid_agent_pos = [(h-2, c) for c in range(1, w-1)]
        agent_valid_positions = list(set(valid_agent_pos))
        # generate valid marker positions
        expected_marker_positions = [(h-2, c) for c in range(1, w-1) if not s[h-2, c, 6]]
        not_expected_marker_positions = [(h-2, c) for c in range(1, w-1) if s[h-2, c, 6]]
        metadata = {'agent_valid_positions':agent_valid_positions,
                    'expected_marker_positions':expected_marker_positions,
                    'not_expected_marker_positions': not_expected_marker_positions}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

    # generate an initial env
    def generate_single_state_stairClimber(self, h=8, w=8, wall_prob=0.1, env_task_metadata={}):
        s = np.zeros([h, w, len(state_table)]) > 0
        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True

        world_map = [
            ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            ['-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-'],
            ['-',   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-', '-'],
            ['-',   0,   0,   0,   0,   0,   0,   0,   0, '-', '-',   0, '-'],
            ['-',   0,   0,   0,   0,   0,   0,   0, '-', '-',   0,   0, '-'],
            ['-',   0,   0,   0,   0,   0,   0, '-', '-',   0,   0,   0, '-'],
            ['-',   0,   0,   0,   0,   0, '-', '-',   0,   0,   0,   0, '-'],
            ['-',   0,   0,   0,   0, '-', '-',   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0,   0, '-', '-',   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0, '-', '-',   0,   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0, '-', '-',   0,   0,   0,   0,   0,   0,   0,   0, '-'],
            ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
        ]

        c = 2
        r = h - 1
        valid_agent_pos = []
        valid_init_pos = []
        while r > 0 and c < w:
            s[r, c, 4] = True
            s[r - 1, c, 4] = True
            if r - 1 > 0 and c - 1 > 0:
                valid_agent_pos.append((r - 1, c - 1))
                valid_init_pos.append((r - 1, c - 1))
                assert not s[r - 1, c - 1, 4] , "there shouldn't be a wall at {}, {}".format(r - 1, c - 1)
            if r - 2 > 0 and c - 1 > 0:
                valid_agent_pos.append((r - 2, c - 1))
                assert not s[r - 2, c - 1, 4], "there shouldn't be a wall at {}, {}".format(r - 2, c - 1)
            if r - 2 > 0 and c > 0:
                valid_agent_pos.append((r - 2, c))
                valid_init_pos.append((r - 2, c))
                assert not s[r - 2, c, 4], "there shouldn't be a wall at {}, {}".format(r - 2, c)
            c += 1
            r -= 1

        agent_valid_positions = list(set(valid_agent_pos))
        valid_init_pos = sorted(list(set(valid_init_pos)), key=lambda x: x[1])

        # Karel initial location
        l1, l2 = 0, 0
        while l1 == l2:
            l1, l2 = self.rng.randint(0, len(valid_init_pos)), self.rng.randint(0, len(valid_init_pos))
        agent_idx, marker_idx = min(l1, l2), max(l1, l2)
        agent_pos, marker_pos = valid_init_pos[agent_idx], valid_init_pos[marker_idx]
        assert (not s[agent_pos[0], agent_pos[1], 4]) and not (s[marker_pos[0], marker_pos[1], 4])
        s[agent_pos[0], agent_pos[1], 1] = True

        # Marker: num of max marker == 1 for now
        s[:, :, 5] = True
        s[marker_pos[0], marker_pos[1], 5] = False
        s[marker_pos[0], marker_pos[1], 6] = True

        assert np.sum(s[:, :, 6]) == 1
        metadata = {'agent_valid_positions': agent_valid_positions}
        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

    # generate an initial env for random maze key2door problem
    def generate_single_state_random_maze_key2door(self, h=8, w=8, wall_prob=0.1, env_task_metadata={}, is_top_off=False):
        """
        initial state generator for random maze problem
        Valid program for random maze problem:
        DEF run m( WHILE c( noMarkersPresent c) w( IFELSE c( rightIsClear c) i( turnRight i) ELSE e( WHILE c( not c( frontIsClear c) c) w( turnLeft w) e) move w) m)

        :param h:
        :param w:
        :param wall_prob:
        :return:
        """
        world_map = [
            ['-', '-', '-', '-', '-', '-', '-', '-'],
            ['-',   0,   0,   0,   0,   0, 'M', '-'],
            ['-', '-', '-',   0, '-', '-', '-', '-'],
            ['-', '-', '-',   0, '-', '-', '-', '-'],
            ['-', '-', '-',   0, '-', '-', '-', '-'],
            ['-', '-', '-',   0, '-', '-', '-', '-'],
            ['-', '-', '-', 'M', '-', '-', '-', '-'],
            ['-', '-', '-', '-', '-', '-', '-', '-'],
        ]

        s = np.zeros([h, w, len(state_table)]) > 0
        init_pos = (1,1)

        for y1 in range(h):
            for x1 in range(w):
                s[y1, x1, 4] = world_map[y1][x1] == '-'
                s[y1, x1, 5] = True if world_map[y1][x1] != 'M' else False
                s[y1, x1, 6] = True if world_map[y1][x1] == 'M' else False

        # init karel agent position
        agent_pos = init_pos
        s[agent_pos[0], agent_pos[1], 1] = True

        '''
        # Marker location
        valid_loc = 0
        while (valid_loc < 2):
           ym = self.rng.randint(0, h)
           xm = self.rng.randint(0, w)
           if (not s[ym, xm, 4]) and (not s[ym, xm, 6]):
               valid_loc += 1
               s[ym, xm, 6] = True
               assert not s[ym, xm, 4]

        assert not s[ym, xm, 4]
        '''

        # put 0 markers everywhere but 2 location
        s[:, :, 5] = 1 - (np.sum(s[:, :, 6:], axis=-1) > 0) > 0
        assert np.sum(s[:, :, 6]) == 2
        metadata = {'agent_valid_positions': None}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata


    # generate an initial env for random maze key2door problem
    def generate_single_state_random_maze_key2doorSpace(self, h=8, w=8, wall_prob=0.1, env_task_metadata={}, is_top_off=False):
        """
        initial state generator for random maze problem
        Valid program for random maze problem:
        DEF run m( WHILE c( noMarkersPresent c) w( IFELSE c( rightIsClear c) i( turnRight i) ELSE e( WHILE c( not c( frontIsClear c) c) w( turnLeft w) e) move w) m)

        :param h:
        :param w:
        :param wall_prob:
        :return:
        """
        world_map = [
            ['-', '-', '-', '-', '-', '-', '-', '-'],
            ['-',   0,   0,   0,   0,   0, 'M', '-'],
            ['-',   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0, 'M',   0,   0,   0, '-'],
            ['-', '-', '-', '-', '-', '-', '-', '-'],
        ]

        s = np.zeros([h, w, len(state_table)]) > 0
        init_pos = (1,1)

        for y1 in range(h):
            for x1 in range(w):
                s[y1, x1, 4] = world_map[y1][x1] == '-'
                s[y1, x1, 5] = True if world_map[y1][x1] != 'M' else False
                s[y1, x1, 6] = True if world_map[y1][x1] == 'M' else False

        # init karel agent position
        agent_pos = init_pos
        s[agent_pos[0], agent_pos[1], 1] = True

        # put 0 markers everywhere but 2 location
        s[:, :, 5] = 1 - (np.sum(s[:, :, 6:], axis=-1) > 0) > 0
        assert np.sum(s[:, :, 6]) == 2
        metadata = {'agent_valid_positions': None}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

    def generate_single_state_oneStroke(self, h=8, w=8, wall_prob=0.1, env_task_metadata={}, is_top_off=False):
        """
        initial state generator for oneStoke problem
        Valid program for harvester problem:
        DEF run m( WHILE c( markersPresent c) w( WHILE c( markersPresent c) w( pickMarker move w) turnRight move turnLeft WHILE c( markersPresent c) w( pickMarker move w) turnLeft move turnRight w) m)

        :param h:
        :param w:
        :param wall_prob:
        :return:
        """

        s = np.zeros([h, w, len(state_table)]) > 0
        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True

        # initial karel position: karel facing east at the last row in environment
        #agent_pos = (h-2, 1)
        agent_pos = np.random.randint(1, h-1, size=[2])
        agent_dir = 1
        s[agent_pos[0], agent_pos[1], agent_dir] = True

        metadata = {}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

    # generate an initial env doorkey problem
    def generate_single_state_doorkey(self, h=8, w=8, wall_prob=0.1, env_task_metadata={}, is_top_off=False):
        """
        initial state generator for doorkey problem
        Valid program for random maze problem:
        TODO
        DEF run m( WHILE c( noMarkersPresent c) w( IFELSE c( rightIsClear c) i( turnRight i) ELSE e( WHILE c( not c( frontIsClear c) c) w( turnLeft w) e) move w) m)

        :param h:
        :param w:
        :param wall_prob:
        :return:
        """

        # world_map = [
        #     ['-', '-', '-', '-', '-', '-', '-', '-'],
        #     ['-',   0,   0,   0, '-',   0,   0, '-'],
        #     ['-',   0,   0,   0, '-',   0,   0, '-'],
        #     ['-',   0,   0,   0, '-',   0,   0, '-'],
        #     ['-',   0,   0,   0, '-',   0,   0, '-'],
        #     ['-',   0, 'M',   0, '-',   0,   0, '-'],
        #     ['-',   0,   0,   0, '-',   0, 'M', '-'],
        #     ['-', '-', '-', '-', '-', '-', '-', '-'],
        # ]
        world_map = [
            ['-', '-', '-', '-', '-', '-', '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0, '-'],
            ['-',   0,   0,   0, '-',   0,   0, '-'],
            ['-',   0,   0,   0, '-',   0,   0, '-'],
            ['-',   0,   0,   0, '-',   0,   0, '-'],
            ['-',   0,   0,   0, '-',   0,   0, '-'],
            ['-',   0,   0,   0, '-',   0,   0, '-'],
            ['-', '-', '-', '-', '-', '-', '-', '-'],
        ]

        s = np.zeros([h, w, len(state_table)]) > 0
        init_pos = (np.random.randint(1, h-1), np.random.randint(1, 4))
        key = (np.random.randint(1, h-1), np.random.randint(1, 4))
        target = (np.random.randint(1, h-1), np.random.randint(5, h-1))
        world_map[key[0]][key[1]] = 'M'
        world_map[target[0]][target[1]] = 'M' 
        # init_pos = (1,1)
        door_pos = [(2,4), (3,4)]

        for y1 in range(h):
            for x1 in range(w):
                s[y1, x1, 4] = world_map[y1][x1] == '-'
                s[y1, x1, 5] = True if world_map[y1][x1] != 'M' else False
                s[y1, x1, 6] = True if world_map[y1][x1] == 'M' else False

        # init karel agent position
        agent_pos = init_pos
        s[agent_pos[0], agent_pos[1], 1] = True

        # put 0 markers everywhere but 2 location
        #s[:, :, 5] = 1 - (np.sum(s[:, :, 6:], axis=-1) > 0) > 0
        assert np.sum(s[:, :, 6]) == 2
        metadata = {'agent_valid_positions': None, 'door_positions': door_pos, 'key': key, 'target': target}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

    def generate_single_state_seeder(self, h=8, w=8, wall_prob=0.1, env_task_metadata={}, is_top_off=False):
        """
        initial state generator for seeder problem
        Valid program for harvester problem:
        DEF run m( WHILE c( noMarkersPresent c) w( WHILE c( noMarkersPresent c) w( putMarker move w) turnRight move turnLeft WHILE c( noMarkersPresent c) w( putMarker move w) turnLeft move turnRight w) m)

        :param h:
        :param w:
        :param wall_prob:
        :return:
        """

        s = np.zeros([h, w, len(state_table)]) > 0
        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True
        
        # predefine a set of placed marker
        existing_marker = []
        #existing_marker = [
        #        (1, 1), (1, 3), (1, 5),
        #        (2, 2), (2, 4), (2, 6),
        #        (3, 1), (3, 3), (3, 5),
        #        (4, 2), (4, 4), (4, 6),
        #        (5, 1), (5, 3), (5, 5),
        #        (6, 2), (6, 4), (6, 6),
        #        ]

        # initial karel position: karel facing east at the last row in environment
        agent_pos = (h-2, 1)
        s[agent_pos[0], agent_pos[1], 1] = True
        
        # agent_pos = np.random.randint(1, h-1, size=[2])
        # agent_dir = np.random.randint(0, 4, size=[1])
        # s[agent_pos[0], agent_pos[1], agent_dir[0]] = True
    
        # place marker
        for m in existing_marker:
            s[m[0], m[1], 6] = True

        metadata = {'existing_marker': existing_marker}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

    def generate_single_state_snake(self, h=8, w=8, wall_prob=0.1, env_task_metadata={}, is_top_off=False):
        """
        initial state generator for snake problem
        Valid program for harvester problem:
        DEF run m( WHILE c( markersPresent c) w( WHILE c( markersPresent c) w( pickMarker move w) turnRight move turnLeft WHILE c( markersPresent c) w( pickMarker move w) turnLeft move turnRight w) m)

        :param h:
        :param w:
        :param wall_prob:
        :return:
        """

        s = np.zeros([h, w, len(state_table)]) > 0
        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True

        # initial karel position: karel facing east at the first row in environment
        agent_pos = (1, 3)
        agent_dir = 1
        s[agent_pos[0], agent_pos[1], agent_dir] = True

        # add initial body of the snake
        s[1, 1, 7] = True
        s[1, 2, 7] = True

        # add first marker
        s[1, 6, 6] = True
        
        marker_list = [
                (6, 6), (6, 1), (1, 1), 
                (2, 2), (2, 5), (5, 5), (5, 2), 
                (3, 3), (5, 3), (4, 4), (6, 4), 
                (3, 2), (1, 4), (6, 2), (2, 6),
                (4, 1), (1, 3), (2, 1), (3, 6),
                (4, 3), (5, 1), (2, 4), (5, 4),
                (1, 5), (3, 5), (4, 6), (5, 6),
                (3, 1), (4, 2), (6, 5), (2, 3),
                (1, 2), (4, 5), (3, 4), (6, 3),
                ]
        marker_pointer = 0

        metadata = {'marker_list': marker_list, 'marker_pointer': marker_pointer}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

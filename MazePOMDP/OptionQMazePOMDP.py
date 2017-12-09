import numpy as np
from gym.spaces import prng
import random
import gym
import time
from option_space import OptionSpace
from pomdp_maze import POMDPMaze

class Agent(object):
    def __init__(self):
        self.episode = 0
        
        self.gamma = 0.8
        self.alpha = 0.5

    def seed(self, seed):
        prng.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def set_environment(self, env):
        self.env = env

        # state space
        NUM_ACTIONS =  4
        NUM_STATES = 7

        # option space
        actions_per_state = list()
        for state in range(NUM_STATES):
            actions_per_state.append(NUM_ACTIONS)
        self.option_space = OptionSpace(actions_per_state, 3)
        
        NUM_OPTIONS = self.option_space.num_options
        
        self.q_table = 0.0 * np.ones((NUM_STATES, NUM_OPTIONS ))

    def train_episode(self):
        # -- setup --
        self.global_step = 0
        s = self.env.reset()
        s_next = s
        o = self.new_policy(s)
        done = False
        option_stack = list()   # save the list of (s,o) pairs; once done
                                # update inversed for compliance with semi-MDP

        # -- execute episode --
        acc_reward = 0
        reward_sequence = list()
        actions = ("left","wait","right")
        alpha = self.alpha
        gamma = self.gamma
        
        options_executed = 0
        while not done:
            option = self.new_policy(s)
            s_next, reward, done, time_steps = self.execute_option(s,option)
            options_executed += 1

            # update option -- smdp-Q-learning on top level
            if not done:
                valid_options = self.option_space.options_in_state(s_next)
                update = reward + ((gamma ** time_steps) *
                                    np.amax(self.q_table[s_next,valid_options]))
            else:
                update = reward
            self.q_table[s, option] += alpha * (update - self.q_table[s,option])

            s = s_next
        print("Episode %d. Steps: %d Options: %d" % (self.episode, self.global_step,
                                                     options_executed))
        #print(np.amax(self.q_table, 1))
        self.episode += 1

    def execute_option(self, s, option):
        alpha = self.alpha
        gamma = self.gamma
        
        done = False
        total_time_steps = 0
        discounted_reward = 0
        while option is not None and not done:
            action = self.option_space.get_action(s,option)
            s_next, reward, done, time_steps = self.execute_action(s,action)

            discounted_reward += (gamma ** total_time_steps) * reward
            total_time_steps += time_steps
            
            option = self.option_space.o_new(s,option)
            s = s_next

        return s_next, discounted_reward, done, time_steps

    def execute_action(self, s, action):
        # the idea is to execute an action until a state change is observed
        # this turns the environment into a partially observable semi-MDP

        if action is None:
            # terminal action
            s_next = s
            reward = 0
            time_steps = 1
            done = False
            return s_next, reward, done, time_steps

        gamma = self.gamma
        discounted_reward = 0
        time_steps = 0
        done = False
        s_next = s
        
        while s == s_next and not done and time_steps < 100:
            s_next, reward, done, _ = self.env.step(action)

            discounted_reward += (gamma ** time_steps) * reward
            time_steps += 1
            self.global_step += 1

        return s_next, discounted_reward, done, time_steps

    def new_policy(self, s):
        exploration_rate = 0.0
        valid_options = self.option_space.options_in_state(s)
        
        if random.random() < exploration_rate:
            return np.random.choice(valid_options,1)[0]
        else:
            option_idx = np.argmax(self.q_table[s,valid_options])
            best_option = valid_options[option_idx]
            return best_option

def main():
    env = POMDPMaze()
    env.seed(500)
    prng.seed(48)
    agent = Agent()
    agent.set_environment(env)
    agent.seed(48)

    for idx in range(2501):
        agent.train_episode()

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 19:42:47 2018

@author: Reet Barik
"""

import math
import time
import copy
import random
import operator
import numpy as np

class GridWorld:

    grid = None
    num_row = None
    num_col = None

    wall_rep = "_"
    pos_rep = "*"
    goal_rep = "+"
    bad_rep = "-"

    def get_cell(self, pos):
        return self.grid[pos[0]][pos[1]]
    
    def __str__(self):
        
        max_width = 0
        for row in self.grid:
            for cell in row:
                if len(cell) > max_width:
                    max_width = len(cell)

        title = "WORLD\n"
        this_str = ""
        sep = "=" * ((len(self.grid) + 4) * max_width) + "\n"
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                pad = (max_width - len(cell) + 1) * " "
                this_str += cell
                this_str += pad
            this_str += "\n"
        this_str = title + sep + this_str + sep
            
        return this_str

    def __init__(self, grid):
        
        self.grid = grid
        self.num_row = len(grid)
        self.num_col = len(grid[0])
        
        

class Policy:

    world = None
    actions = None
    pos = None 
    start_pos = None
    goal_pos = None

    discount_factor = None
    learning_rate = None
    q_matrix = None
    rew_matrix = None
    
    def reset(self):

        self.pos = copy.deepcopy(self.start_pos)
        

    def next(self):        
        raise NotImplementedError
        

    def action_to_pos(self, action):
        
        pos = copy.deepcopy(self.pos)
        if action == self.actions[0]:
            if pos[0] != 0: pos[0] -= 1
        elif action == self.actions[1]:
            if pos[1] != (self.world.num_col - 1): pos[1] += 1
        elif action == self.actions[2]:
            if pos[0] != (self.world.num_row - 1): pos[0] += 1
        elif action == self.actions[3]:
            if pos[1] != 0: pos[1] -= 1
        else:
            raise RuntimeError

        if self.world.get_cell(pos) == GridWorld.wall_rep:
            pos = self.pos

        return pos
    

    def get_best_action(self, pos):
        
        pos_index = pos[0] * self.world.num_col + pos[1]
        moves = self.q_matrix[pos_index]
        
        action_dict = {}
        for move_i, move_q in enumerate(moves):
            action_dict[move_i] = move_q
        action_items = list(action_dict.items())
        random.shuffle(action_items)

        highest_move = -1
        highest_move_index = -1
        for move_i, move_q in action_items:
            if move_q > highest_move:
                highest_move = move_q
                highest_move_index = move_i

        return highest_move_index, highest_move
    

    def get_Q_matrix_pos(self, pos):        
        return pos[0] * self.world.num_col + pos[1]
    

    def get_reward(self, pos):
        
        c = self.world.grid[pos[0]][pos[1]]
        self.rew_matrix[pos[0]][pos[1]] = float(c)

        return float(c)
    

    def __str__(self):
        
        action_sym = ['^', '>', 'v', '<']

        title = "POLICY\n"
        pad_len = 3
        this_str = ""
        sep = "=" * ((len(self.world.grid) + pad_len + 1) * 2) + "\n"
    
        for i, row in enumerate(self.world.grid):
            for j, cell in enumerate(row):

                new_elem = ""

                if self.pos == [i, j]:
                    new_elem = "*"
                if cell == GridWorld.wall_rep:
                    new_elem = " "
                elif [i, j] == self.goal_pos:
                    new_elem += GridWorld.goal_rep
                elif self.world.get_cell([i, j]) == "-1":
                    new_elem += GridWorld.bad_rep
                else:
                    a_i, a_q = self.get_best_action([i, j])
                    new_elem += action_sym[a_i]

                new_elem += " " * (pad_len - len(new_elem))
                this_str += new_elem
            this_str += "\n"
        this_str = title + sep + this_str + sep
            
        return this_str
    

    def __init__(self, world, start_pos, goal_pos, discount_factor,
                 learning_rate):
        
        actions = ['u', 'r', 'd', 'l']
        num_states = world.num_row * world.num_col
        num_actions = len(actions)

        self.world = world
        self.start_pos = copy.deepcopy(start_pos)
        self.goal_pos = copy.deepcopy(goal_pos)
        self.pos = copy.deepcopy(start_pos)
        self.actions = actions
        self.q_matrix = np.zeros((num_states, num_actions))
        self.rew_matrix = np.zeros((world.num_row, world.num_col))
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        
        

class EpsilonGreedyPolicy(Policy):
    
    epsilon = None

    def next(self):
        
        orig_state = copy.deepcopy(self.pos)    # state s
        pos_index = self.get_Q_matrix_pos(self.pos)
        moves = self.q_matrix[pos_index]

        highest_move_index, highest_move_q = self.get_best_action(
            orig_state)

        rand_val = random.uniform(0, 1)
        if rand_val < self.epsilon: picked_move = random.randint(0, 3)
        else: picked_move = highest_move_index

        action = self.actions[picked_move]    
        
        self.pos = self.action_to_pos(action)  

        reward = self.get_reward(self.pos)              
        
        opt_future_i, opt_future_q = self.get_best_action(self.pos)
        
        this_q = self.q_matrix[pos_index][picked_move]  
        x = self.learning_rate * (reward + self.discount_factor * \
                                  opt_future_q - this_q)
        self.q_matrix[pos_index][picked_move] += x

        if self.pos == self.goal_pos: return False
        else: return True
        
    
    def __init__(self, world, start_pos, goal_pos, discount_factor,
                 learning_rate, epsilon):
        Policy.__init__(self, world, start_pos, goal_pos, discount_factor,
                        learning_rate)
        self.epsilon = epsilon
        
        

class BoltzmannExplorationPolicy(Policy):

    temperature = None

    def reset(self):
               
        if self.temperature >= 1: self.temperature -= 1
        Policy.reset(self)
        

    def next(self):

        orig_state = copy.deepcopy(self.pos)    # state s
        pos_index = self.get_Q_matrix_pos(self.pos)
        moves = self.q_matrix[pos_index]

        if self.temperature > 0:
            action_probs_numes = []
            denom = 0
            for m in moves:
                val = math.exp(m / self.temperature)
                action_probs_numes.append(val)
                denom += val
            action_probs = [x / denom for x in action_probs_numes]

            rand_val = random.uniform(0, 1)
            prob_sum = 0
            for i, prob in enumerate(action_probs):
                prob_sum += prob
                if rand_val <= prob_sum:
                    picked_move = i
                    break
        else:
            picked_move, picked_move_q = self.get_best_action(orig_state)

        action = self.actions[picked_move]      
        self.pos = self.action_to_pos(action)   

        reward = self.get_reward(self.pos)             
        opt_future_i, opt_future_q = self.get_best_action(self.pos)
        this_q = self.q_matrix[pos_index][picked_move]  
        x = self.learning_rate * (reward + self.discount_factor * \
                                  opt_future_q - this_q)
        self.q_matrix[pos_index][picked_move] += x

        if self.pos == self.goal_pos: return False
        else: return True
        

    def __str__(self):
        orig_state = copy.deepcopy(self.pos)
        pos_index = self.get_Q_matrix_pos(self.pos)
        moves = self.q_matrix[pos_index]
        action_probs_numes = []
        denom = 0
        for m in moves:
            val = math.exp(m / self.temperature)
            action_probs_numes.append(val)
            denom += val
        action_probs = [x / denom for x in action_probs_numes]
        
        this_str = Policy.__str__(self)
        return this_str
    
    
    def __init__(self, world, start_pos, goal_pos, discount_factor,
                 learning_rate, temperature):
        Policy.__init__(self, world, start_pos, goal_pos, discount_factor,
                        learning_rate)
        self.temperature = temperature
        


def print_policy(policy):
    print("Printing policy:")
    print()
    print(policy)
    return True



def print_policy_q_matrix(policy):
    print("Printing Q values:")
    print()
    print(policy.q_matrix)
    return



def printq(policy, param, is_epsilon = True):
    if(is_epsilon):
        print()
        print("Policy: EpsilonGreedy, Epsilon = " + str(param))
        print()
        print_policy(policy)
        print()
        print_policy_q_matrix(policy)
    else: 
        print()
        print("Policy: BoltzmanExploration, Temperature = " + str(param))
        print()
        print_policy(policy)
        print()
        print_policy_q_matrix(policy)
    return



def rep_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False


def matrix_diff(matrix_one, matrix_two):
    total_diff = 0
    for i, row in enumerate(matrix_one):
        index_one, value_one = max(enumerate(matrix_one[i]),
                                   key=operator.itemgetter(1))
        index_two, value_two = max(enumerate(matrix_two[i]),
                                   key=operator.itemgetter(1))
        if index_one != index_two: total_diff += 1
        
    return total_diff        



def create_epsilon_greedy_policy(world, epsilon):
    
    discount_factor = 0.9
    learning_rate = 0.01
    start_pos = [0, 0]  
    goal_pos = [5, 5]  

    return EpsilonGreedyPolicy(world, start_pos, goal_pos,
                               discount_factor, learning_rate, epsilon)
    
    

def create_boltzmann_policy(world, temperature):
    
    discount_factor = 0.9
    learning_rate = 0.01
    start_pos = [0, 0] 
    goal_pos = [5, 5]   

    return BoltzmannExplorationPolicy(world, start_pos, goal_pos,
                               discount_factor, learning_rate, temperature)
    
    

def create_world():
    
    num_cols = 10
    num_rows = 10
    grid = [['0' for col in range(num_cols)]
             for row in range(num_rows)]

    for i in range(1, 5): grid[2][i] = GridWorld.wall_rep
    for i in range(6, 9): grid[2][i] = GridWorld.wall_rep
    for i in range(3, 8): grid[i][4] = GridWorld.wall_rep
    
    grid[5][5] = '1'

    grid[4][5] = '-1'
    grid[4][6] = '-1'
    grid[5][6] = '-1'
    grid[5][8] = '-1'
    grid[6][8] = '-1'
    grid[7][3] = '-1'
    grid[7][5] = '-1'
    grid[7][6] = '-1'
    
    return GridWorld(grid)



if __name__ == "__main__":

    print("\nQ9: Q-learning on Gridworld \n")
    
    world = create_world()
    policies = []
        
    policy_funcs = [create_epsilon_greedy_policy,
                    create_boltzmann_policy]
    policy_params = [("epsilon", [0.1, 0.2, 0.3]),
                     ("temperature", [1000, 800, 600, 500, 250, 100, 10, 1])]
    
    print()
    print(world)

    print("Running two different explore/exploit policies:")
    print(" A. Epsilon-greedy,", "epsilon =",
          policy_params[0][1])
    print(" B. Boltzmann exploration,", "temperature =",
          policy_params[1][1])
    print()

    print("Doing Q-learning:")
    for i, policy_func in enumerate(policy_funcs):
        param_name, param_list = policy_params[i]
        for param in param_list:
            policy = policy_func(world, param)
            print(str(len(policies) + 1) + ".", "Running",
                  policy.__class__.__name__ + ",", param_name,
                  "=", param)
            
            ts = time.clock()
            num_iter = 0
            this_conv_count = 1000
            
            while True:
                num_iter += 1
                last_q_matrix = copy.deepcopy(policy.q_matrix)

                while policy.next() == True: pass
                policy.reset()

                if matrix_diff(last_q_matrix, policy.q_matrix) \
                   <= 0:
                    this_conv_count -= 1
                    if this_conv_count == 0: break
                else: this_conv_count = 1000
            
            te = time.clock()
            print(("  Number of iterations required to converge: "), str((num_iter - 1000)))

            policies.append(policy)
            
    print()
    print("The converged Q values for each state-action pair for each of the above policies are as follows: \n")

    for i in range(11):
        if(i < 3):
            printq(policies[i], policy_params[0][1][i])
        else:
            printq(policies[i], policy_params[1][1][i - 3], False)
    print()
    print("______________________________________________________________________________________________________________________________________________________________________________________________________")
import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
from collections import deque
from copy import deepcopy
import time
from IPython.display import clear_output
import heapdict
from math import dist



class Node:
    def __init__(self, state, total_cost, actions, env, is_hole = False):
        self.state = state
        self.total_cost = total_cost
        self.actions = actions
        self.env = env
        self.is_hole = is_hole



class BFSAgent:
    def __init__(self) -> None:
        self.env = None

    def animation(self, epochs: int ,state: int, action: List[int], total_cost: int) -> None:
        clear_output(wait=True)
        print(self.env.render())
        print(f"Timestep: {epochs}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Total Cost: {total_cost}")
        time.sleep(1)

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        initial_state = self.env.get_initial_state()
        expansions = 0
        node = Node(initial_state, 0, [], env)
        if env.is_final_state(node.state):
            return (node.actions, node.total_cost, expansions)
        open_list = deque([node])
        closed_list = []

        while open_list:
            node = open_list.pop()
            closed_list += [node.state]
            expansions += 1
            if node.is_hole:
                continue
            for action, successor in env.succ(node.state).items():
                env_copy = deepcopy(node.env)
                new_state, cost, terminated = env_copy.step(action)
                is_hole = terminated == True and not env.is_final_state(new_state)
                child_node = Node(new_state, node.total_cost + cost, node.actions + [action], env_copy, is_hole)
                if terminated and not env_copy.is_final_state(child_node.state) and child_node.state not in closed_list:
                    open_list.appendleft(child_node)
                    closed_list += [child_node.state]
                elif child_node.state not in closed_list and not any(node.state == child_node.state for node in open_list):
                    if env.is_final_state(child_node.state):
                        return (child_node.actions, child_node.total_cost, expansions)
                    open_list.appendleft(child_node)
                    #closed_list += [child_node.state]


        return (None, None, expansions)
    

class Node2:
    def __init__(self, state, env, actions=[], g = 0, h_weight = 0):
        self.state = state      
        self.actions = actions
        self.env = env
        self.g = g
        self.f = self.h_MSAP(state, env) * h_weight + (1-h_weight) * g
        
    def manhatan(self, x1, y1, x2, y2):
        return abs(x1-x2) + abs(y1-y2)
    
    def euclidean(self, x1, y1, x2, y2):
        p1 = [x1,y1]
        p2 = [x2,y2]
        return dist(p1,p2)

    def h_MSAP(self, state, env):
        x,y = env.to_row_col(state)
        xg, yg = env.to_row_col(env.goals[0])
        xd1, yd1 = env.to_row_col(env.d1)
        xd2, yd2 = env.to_row_col(env.d2)
        if not env.collected_dragon_balls[0] and not env.collected_dragon_balls[1]:
            return min(self.manhatan(x,y,xg,yg), self.manhatan(x,y,xd1,yd1), self.manhatan(x,y,xd2,yd2))
        elif not env.collected_dragon_balls[1] and env.collected_dragon_balls[0]:
            return min(self.manhatan(x,y,xg,yg), self.manhatan(x,y,xd2,yd2))
        elif not env.collected_dragon_balls[0] and env.collected_dragon_balls[1]:
            return min(self.manhatan(x,y,xg,yg), self.manhatan(x,y,xd1,yd1))
        else:
            return self.manhatan(x,y,xg,yg)


    def __lt__(self, other):
        if self.f < other.f:
            return True
        elif self.f == other.f:
            return self.state[0] < other.state[0]
        return False




class WeightedAStarAgent:

    def __init__(self):
        self.env = None

    def search(self, env, h_weight) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        expanded_counter = 0

        node = Node2(self.env.get_initial_state(), env, [], 0, h_weight)
        open_queue = heapdict.heapdict()
        open_queue[node.state] = node
        closed_queue = {}

        while open_queue:
            _, node = open_queue.popitem()
            closed_queue[node.state] = node

            if env.is_final_state(node.state):
                return (node.actions, node.g, expanded_counter)

            expanded_counter += 1

            for action, successor in env.succ(node.state).items():
                env_copy = deepcopy(node.env)
                new_state, cost, terminated = env_copy.step(action)
                child = Node2(new_state, env_copy, node.actions + [action], node.g + cost, h_weight)
                if new_state not in closed_queue and new_state not in open_queue:
                    open_queue[child.state] = child

                elif new_state in open_queue:
                    curr_f = open_queue[new_state].f
                    if child.f < curr_f:
                        open_queue[new_state] = child
                
                elif new_state in closed_queue:
                    curr_f = closed_queue[new_state].f
                    if child.f < curr_f:
                        open_queue[new_state] = child
                        del closed_queue[new_state]

        return (None, None, expanded_counter)







class AStarEpsilonAgent:
    def __init__(self):
        self.env = None

    def search(self, env, epsilon) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        expanded_counter = 0

        node = Node2(self.env.get_initial_state(), env, [], 0, 0)
        open_queue = heapdict.heapdict()
        open_queue[node.state] = node
        closed_set = set()
        focal_list = heapdict.heapdict()
        focal_list[node.state] = node
        min_f = float('inf')

        while open_queue:
            focal_list.clear()
            min_f = min((n.f for n in open_queue.values()), default=float('inf'))
            for state, node in open_queue.items():
                if node.f <= (1 + epsilon) * min_f:
                    focal_list[state] = node

            if not focal_list:
                break

            _, focal_node = focal_list.popitem()
            closed_set.add(focal_node.state)
            expanded_counter += 1

            if env.is_final_state(focal_node.state):
                return (focal_node.actions, focal_node.g, expanded_counter)

            #print(len(focal_list))
            #print(min_f)

            for action, successor in env.succ(focal_node.state).items():
                env_copy = deepcopy(focal_node.env)
                successor_state, cost, terminated = env_copy.step(action)
                #print(successor_state)
                if successor_state in closed_set:
                    continue

                child_node = Node2(successor_state, env_copy, focal_node.actions + [action], focal_node.g + cost, 0)

                if successor_state not in open_queue or child_node.f < open_queue[successor_state].f:
                    open_queue[successor_state] = child_node

                elif successor_state in open_queue:
                    curr_f = open_queue[successor_state].f
                    if child_node.f < curr_f:
                        open_queue[successor_state] = child_node
                
                elif successor_state in closed_set:
                    curr_f = closed_set[successor_state].f
                    if child_node.f < curr_f:
                        open_queue[successor_state] = child_node
                        del closed_set[successor_state]
            open_queue.popitem()


        return ([], float('inf'), expanded_counter)

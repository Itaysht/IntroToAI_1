from queue import SimpleQueue

import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict

class Node:
    def __init__(self, env, state, father, op, term):
        self.env = env
        self.state = state
        self.operator = op
        self.father = father
        self.terminate = term
    def expand(self):
        #return list(set([(i, self.env.succ(self.state)[i]) for i in range(3)]))
        lst = []
        for act, (news, cos, term) in self.env.succ(self.state).items():
            news = (news[0], self.state[1], self.state[2])
            lst += [(act, (news, cos, term))]
        return lst
        #return list(set([s for s in self.env.succ(self.state).items()]))

class BFSAgent():
    def __init__(self) -> None:
        pass

    def exist_in_queue(self, queue, node, length):
        for i in range(length):
            if (queue[i].state[0] == node.state[0]) and (queue[i].state[1] == node.state[1]) and (queue[i].state[2] == node.state[2]):
                return True
        return False

    def rev_path(self, node):
        going_up_node = node
        result_actions = []
        while (going_up_node.father is not None):
            result_actions += [going_up_node.operator]
            going_up_node = going_up_node.father
        cost = 0
        going_up_node.env.reset()
        result_actions.reverse()
        for act in result_actions:
            (news, newcost, newterm) = going_up_node.env.step(act)
            cost += newcost
        return result_actions, cost

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        total_cost = 0
        expended = 0

        length = 0
        env.reset()
        node = Node(env, env.get_initial_state(), None, None, False)
        if env.is_final_state(node.state):
            return [], total_cost, expended

        opened_nodes = [node]
        length += 1
        closed_nodes = set()
        while not (length == 0):
            node = opened_nodes.pop(0)
            length -= 1
            if node.terminate and (not env.is_final_state(node.state)):    #if a hole
                continue
            closed_nodes.add(node.state)
            expended += 1
            for act, (stat, cos, term) in node.expand():
                if stat == node.state:       #when the operation is going to the same place
                    continue
                if stat is None:             #when it's a hole
                    continue
                child = Node(env, stat, node, act, term)   #creating the child node
                    #now, check if already developed, or already in queue
                if (stat not in closed_nodes) and (not self.exist_in_queue(opened_nodes, child, length)):
                    if env.is_final_state(child.state):
                        result_actions, total_cost = self.rev_path(child)
                        return result_actions, total_cost, expended
                    if child.state[0] == env.d1[0]:  # check the place of the dragon ball
                        child.state = (env.d1[0], True, child.state[2])
                    if child.state[0] == env.d2[0]:
                        child.state = (env.d2[0], child.state[1], True)
                    opened_nodes.append(child)
                    length += 1


class WeightedAStarAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError



class AStarEpsilonAgent():
    def __init__(self) -> None:
        raise NotImplementedError
        
    def ssearch(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError
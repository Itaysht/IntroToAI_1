from queue import SimpleQueue

import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict


class Node:
    def __init__(self, env, state, father, op, term):
        self.valueF = None
        self.valueG = None
        self.env = env
        self.state = state
        self.operator = op
        self.father = father
        self.terminate = term

    def defineF(self, value):
        self.valueF = value
    def defineG(self, value):
        self.valueG = value

    def foundNewDad(self, nFather, nOp):
        self.operator = nOp
        self.father = nFather

    def expand(self):
        lst = []
        for act, (news, cos, term) in self.env.succ(self.state).items():
            if news is None:
                continue
            news = (news[0], self.state[1], self.state[2])
            lst += [(act, (news, cos, term))]
        return lst

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return (self.state[0] == other.state[0]) and (self.state[1] == other.state[1]) and (self.state[2] == other.state[2])

    def __lt__(self, other):
        return (self.valueF < other.valueF) or ((self.valueF == other.valueF) and (self.state[0] < other.state[0]))


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
            closed_nodes.add(node)
            expended += 1
            if node.terminate:
                continue
            for act, (stat, cos, term) in node.expand():
                if stat == node.state:       #when the operation is going to the same place
                    continue
                child = Node(env, stat, node, act, term)   #creating the child node
                    #now, check if already developed, or already in queue
                if (child not in closed_nodes) and (child not in opened_nodes):
                    if env.is_final_state(child.state):
                        result_actions, total_cost = self.rev_path(child)
                        return result_actions, total_cost, expended
                    if child.state[0] == env.d1[0]:  # check the place of the dragon ball
                        child.state = (env.d1[0], True, child.state[2])
                    if child.state[0] == env.d2[0]:
                        child.state = (env.d2[0], child.state[1], True)
                    if (child not in closed_nodes) and (child not in opened_nodes):
                        opened_nodes.append(child)
                        length += 1
        return [], 0, 0



class WeightedAStarAgent():
    def __init__(self) -> None:
        self.env = None


    def get_from_closed(self, closed, node):
        for i in closed:
            if (i.state[0] == node.state[0]) and (i.state[1] == node.state[1]) and (i.state[2] == node.state[2]):
                return i

    def manhaten(self, stateOne, stateTwo):
        row1, col1 = self.env.to_row_col(stateOne)
        row2, col2 = self.env.to_row_col(stateTwo)
        return abs(row1 - row2) + abs(col1 - col2)

    def computeH(self, d1, d2, goals, state):
        if state[1] and state[2]:
            return min([self.manhaten(i, state) for i in goals])
        if state[1]:
            return min([self.manhaten(d2, state)] + [self.manhaten(i, state) for i in goals if (not (i == state))])
        if state[2]:
            return min([self.manhaten(d1, state)] + [self.manhaten(i, state) for i in goals if (not (i == state))])
        return min([self.manhaten(d1, state), self.manhaten(d2, state)] + [self.manhaten(i, state) for i in goals])

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


    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        total_cost = 0
        expended = 0
        length = 0
        env.reset()
        self.env = env
        node = Node(env, env.get_initial_state(), None, None, False)
        if env.is_final_state(node.state):
            return [], total_cost, expended

        opened_nodes = heapdict.heapdict()
        firstF = self.computeH(env.d1, env.d2, env.get_goal_states(), node.state) * h_weight
        node.defineF(firstF)
        node.defineG(0)
        opened_nodes[node] = (firstF, node.state[0])
        length += 1
        closed_nodes = set()
        while length > 0:
            current_node = opened_nodes.popitem()[0]
            length -= 1
            if env.is_final_state(current_node.state):
                result_actions, total_cost = self.rev_path(current_node)
                return result_actions, total_cost, expended
            closed_nodes.add(current_node)
            expended += 1
            if current_node.terminate:
                continue
            for act, (stat, cos, term) in current_node.expand():
                son_of_current_node = Node(env, stat, current_node, act, term)

                if son_of_current_node.state[0] == env.d1[0]:  # check the place of the dragon ball
                    son_of_current_node.state = (env.d1[0], True, son_of_current_node.state[2])
                if son_of_current_node.state[0] == env.d2[0]:
                    son_of_current_node.state = (env.d2[0], son_of_current_node.state[1], True)

                valueH = self.computeH(env.d1, env.d2, env.get_goal_states(), son_of_current_node.state)
                valueF = (valueH * h_weight) + ((current_node.valueG + cos) * (1 - h_weight))

                if (son_of_current_node not in opened_nodes.keys()) and (son_of_current_node not in closed_nodes):
                    opened_nodes[son_of_current_node] = (valueF, son_of_current_node.state[0])
                    son_of_current_node.defineF(valueF)
                    son_of_current_node.defineG(current_node.valueG + cos)
                    length += 1
                elif son_of_current_node in opened_nodes.keys():
                    original_node = self.get_from_closed(opened_nodes.keys(), son_of_current_node)
                    if current_node.valueG + cos < original_node.valueG:
                        original_node.defineG(current_node.valueG + cos)
                        original_node.foundNewDad(current_node, act)
                    lastF = opened_nodes[son_of_current_node]
                    if valueF < lastF[0]:
                        opened_nodes[original_node] = (valueF, son_of_current_node.state[0])
                        original_node.defineF(valueF)

                else:
                    already_closed = self.get_from_closed(closed_nodes, son_of_current_node)
                    if current_node.valueG + cos < already_closed.valueG:
                        already_closed.defineG(current_node.valueG + cos)
                        already_closed.foundNewDad(current_node, act)
                    if valueF < already_closed.valueF:
                        already_closed.defineF(valueF)
                        opened_nodes[already_closed] = (valueF, already_closed.state[0])
                        length += 1
                        closed_nodes.remove(already_closed)
        return [], 0, 0






class AStarEpsilonAgent():
    def __init__(self) -> None:
        self.env = None

    def get_from_closed(self, closed, node):
        for i in closed:
            if (i.state[0] == node.state[0]) and (i.state[1] == node.state[1]) and (i.state[2] == node.state[2]):
                return i

    def manhaten(self, stateOne, stateTwo):
        row1, col1 = self.env.to_row_col(stateOne)
        row2, col2 = self.env.to_row_col(stateTwo)
        return abs(row1 - row2) + abs(col1 - col2)

    def computeH(self, d1, d2, goals, state):
        if state[1] and state[2]:
            return min([self.manhaten(i, state) for i in goals])
        if state[1]:
            return min([self.manhaten(d2, state)] + [self.manhaten(i, state) for i in goals if (not (i == state))])
        if state[2]:
            return min([self.manhaten(d1, state)] + [self.manhaten(i, state) for i in goals if (not (i == state))])
        return min([self.manhaten(d1, state), self.manhaten(d2, state)] + [self.manhaten(i, state) for i in goals])

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


    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        total_cost = 0
        expended = 0
        length = 0
        env.reset()
        self.env = env
        node = Node(env, env.get_initial_state(), None, None, False)
        if env.is_final_state(node.state):
            return [], total_cost, expended
        opened_nodes = heapdict.heapdict()
        firstF = self.computeH(env.d1, env.d2, env.get_goal_states(), node.state)
        node.defineF(firstF)
        node.defineG(0)
        opened_nodes[node] = (firstF, node.state[0])
        length += 1
        closed_nodes = set()
        while length > 0:
            current_node_value = opened_nodes.peekitem()
            current_node = current_node_value[0]
            current_f_and_t = current_node_value[1]

            focal = [rightNode for rightNode in opened_nodes.keys()
                     if opened_nodes[rightNode][0] <= ((1 + epsilon) * current_f_and_t[0])]
            picked_node = current_node
            g_value_min = current_node.valueG
            index_min = current_node.state[0]
            for foc in focal:
                temp_g_value = foc.valueG
                if (temp_g_value < g_value_min) or ((temp_g_value == g_value_min) and (foc.state[0] < index_min)):
                    g_value_min = temp_g_value
                    picked_node = foc
                    index_min = foc.state[0]

            opened_nodes.pop(picked_node)
            length -= 1

            if env.is_final_state(picked_node.state):
                result_actions, total_cost = self.rev_path(picked_node)
                return result_actions, total_cost, expended

            closed_nodes.add(picked_node)
            expended += 1
            if picked_node.terminate:
                continue
            for act, (stat, cos, term) in picked_node.expand():
                son_of_current_node = Node(env, stat, picked_node, act, term)

                if son_of_current_node.state[0] == env.d1[0]:  # check the place of the dragon ball
                    son_of_current_node.state = (env.d1[0], True, son_of_current_node.state[2])
                if son_of_current_node.state[0] == env.d2[0]:
                    son_of_current_node.state = (env.d2[0], son_of_current_node.state[1], True)

                valueH = self.computeH(env.d1, env.d2, env.get_goal_states(), son_of_current_node.state)
                valueF = valueH + picked_node.valueG + cos

                if (son_of_current_node not in opened_nodes.keys()) and (son_of_current_node not in closed_nodes):
                    opened_nodes[son_of_current_node] = (valueF, son_of_current_node.state[0])
                    son_of_current_node.defineF(valueF)
                    son_of_current_node.defineG(picked_node.valueG + cos)
                    length += 1
                elif son_of_current_node in opened_nodes.keys():
                    lastF = opened_nodes[son_of_current_node]
                    original_node = self.get_from_closed(opened_nodes.keys(), son_of_current_node)
                    if picked_node.valueG + cos < original_node.valueG:
                        original_node.defineG(picked_node.valueG + cos)
                        original_node.foundNewDad(picked_node, act)
                    if valueF < lastF[0]:
                        opened_nodes[original_node] = (valueF, son_of_current_node.state[0])
                        original_node.defineF(valueF)

                else:
                    already_closed = self.get_from_closed(closed_nodes, son_of_current_node)
                    if picked_node.valueG + cos < already_closed.valueG:
                        already_closed.defineG(picked_node.valueG + cos)
                        already_closed.foundNewDad(picked_node, act)
                    if valueF < already_closed.valueF:
                        already_closed.defineF(valueF)
                        opened_nodes[already_closed] = (valueF, already_closed.state[0])
                        length += 1
                        closed_nodes.remove(already_closed)
        return [], 0, 0

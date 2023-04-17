# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import numpy as np
from game import Directions
from util import Stack, Queue, PriorityQueueWithFunction
from functools import partial
s = Directions.SOUTH
w = Directions.WEST
e = Directions.EAST
n = Directions.NORTH

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """

    return  [s, s, w, s, w, w, s, w]


def Graph_Search(problem: SearchProblem, fringe, heuristic = None):

    '''
Persudo Code
    Closed = empty set 
    fringe = SearchDataStructure()
    Init fringe

    while(True):
        if fringe is empty:
            return Failure
        node = fringe.pop()
        if problem.isGoalState( STATE[node] ):
            return node
        if STATE[node] is not in Closed:
            # 加到遍历过的节点中
            add STATE[node] to Closed
            # 所有的邻接节点
            for child-node in EXPAND(STATE[node]):
                fringe.push(child-node)
    '''


    # problem.getSuccessors： [((5, 4), 'South', 1), ((4, 5), 'West', 1)]
    # 解释一下伪代码就是，node可以理解为当前的坐标，包含了两个信息，一个是迄今为止的路径
    # 第二个是当前的位置有没有被走过，涉及到回溯等操作
    # 也就是说放到stack或queue里面的其实是个tuple形式的元素，不妨就令第一个元素是坐标，第二个元素是存的是当前存的路径，
    # 第三个元素是当前节点的步数cost, 第四个元素表示用于排序的代价
    fringe.push( (problem.getStartState(), [], 0, 0)) # maybe change
    closed = []

    while(True):
        if fringe.isEmpty():
            return [Directions.STOP]
        
        node = fringe.pop()
        
        if problem.isGoalState(node[0]):
            return node[1]
        # print(node[0],node[1])

        if node[0] not in closed:
            closed.append(node[0])
            # print("parent:", node[0])
            for childnode in problem.getSuccessors(node[0]):


                path = node[1].copy()
                total_cost = node[2]
                path.append(childnode[1])

                if(heuristic != None):
                    total_cost += heuristic(childnode[0], problem)
                                    
                new_node = (childnode[0], path, node[2] + childnode[2], total_cost + childnode[2])
                # print(childnode[0],path)

                fringe.push(new_node)


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"


    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    stack = Stack()

    return Graph_Search(problem, stack)




def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # 记录踩的坑：append()函数是会修改自身的，并且返回None，所以根本不能用变量去接收它
    # 以及要push list的copy，而不是list本身
    # 没细看指导书就直接写了，才发现是有模板的，重写一遍😥


    queue = Queue()

    return Graph_Search(problem, queue)
                
    # if problem.isGoalState(problem.getStartState()):
    #     return []

    # map = problem.walls.data
    # length = len(map)
    # width = len(map[0])
    # queue = Queue()
    # queue.push(problem.getStartState())
    # dx = [-1, 0, 1, 0]
    # dy = [0, 1, 0, -1]
    # dirs = [w, n, e, s]
    # st = np.zeros((length, width), dtype = np.bool)
    # paths = Queue()
    # paths.push([])

    # while(not queue.isEmpty()):
    #     head = queue.pop()
    #     path = paths.pop()

    #     for i in range(4):
    #         x = head[0] + dx[i]
    #         y = head[1] + dy[i]

    #         if(x >=1 and y >=1 and x <= length - 2 and y <= width - 2 and not map[x][y] and not st[x][y]):
    #             queue.push((x, y))
    #             st[x][y] = True
    #             path.append(dirs[i])
    #             paths.push(path.copy())
    #             if problem.isGoalState((x, y)):
    #                 return path
    #             path.pop()


def uniformFunction(item):
    return item[3]


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    queue = PriorityQueueWithFunction(uniformFunction)
    return Graph_Search(problem, queue)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    queue = PriorityQueueWithFunction(uniformFunction) 
    return Graph_Search(problem, queue, heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

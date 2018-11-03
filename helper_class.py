import copy
from lexicon_pools import LI
'''
REVISION HISTORY
31AUG18 Corrected bugs in BFS, build_level_dict
28AUG18 Initial implementation 
'''
#   these classes and methods are used to assist doAutoLoad() in processing the graph and generating the pools

__author__ = 'Andy Valenti and Jasmine Falk'
__copyright__ = "Copyright 2018. Tufts University"


#   PURPOSE:    A queue can be implemented using python list where the insert() and pop() methods are used to add and
#               remove elements.
#   INPUT: enqueue takes a value to be placed at the tail of the queue; dequeue returns the element at the head.

#   ATTRIBUTION: https://www.tutorialspoint.com/python/python_queue.htm
class Queue:

    def __init__(self):
        self.queue = list()

    def enqueue(self, dataval):

        # Insert method to add element
        if dataval not in self.queue:
            self.queue.insert(0, dataval)
            return True
        return False

    # Pop method to remove element
    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.pop()
        return None

    # test for queue empty
    def isempty(self):
        if len(self.queue) == 0:
            return True
        return False


#   This procedure builds a breadth-first tree as it searches the graph.  The tree corresponds to the pred attributes
#   and is the predecessor subgraph of G. It contains the shortest path from s to each vertex in the tree.
#   The vertex also contains the dist from the root which will be useful when generating the self-inhibitory connections
#   at a given level.
#
#   PURPOSE:    Build a breadh-first tree from G starting at vertex s
#   INPUT:      G as an adjacency list, source as the string name of the vertex which will serve as the root.
#               clear = True will reset all the vertices in the graph, including any "d" values that was set when BFS
#               was run using another source vertex
#   POST-CONDITION: The vertex.pred variables form the predecessor subgraph which is the BFS tree.
#                   The vertex.d variable contains the distance from the root
#                   Returns a list of the instances of the nodes visited
#   ATTRIBUTION: Corman, Leiserson, Riverst & Stein. Introduction to Algorithms, Third Edition. p595-601.

def BFS(g, source, clear=False):
    nodes_visited = []
    Q = Queue()
    s = g.dict_vertices[source]
    # bfs_vertices = copy.deepcopy(g.dict_vertices)
    #
    # bfs_vertices.pop(source, None)       # remove key from dict
    if clear:
        for name, u in g.dict_vertices.iteritems():
            u.color = 0
            u.d = 1000000
            u.pred = None
    s.color = 1
    s.d = 0
    s.pred = None

    Q.enqueue(s)
    while not Q.isempty():
        u = Q.dequeue()
        for v_name in u.neighbors:
            v = g.dict_vertices[v_name]      # convert str to instance
            if v.color == 0:
                v.color = 1
                v.d = u.d + 1
                v.pred = u
                Q.enqueue(v)
        u.color = 2
        nodes_visited.append(u)
    del Q
    return nodes_visited


#   PURPOSE:    Build a dictionary object of vertices at every level. Used to construct self-inhibitory list
#   INPUT:      graph g
#   POST-CONDITION  returns level dictionary
def build_level_dict(g):
    level_dict = {}
    s = g.dict_vertices
    for name, u in s.iteritems():
        if u.d in level_dict:
            level_dict[u.d].append([[u.name[0], 0], LI])
        else:
            level_dict[u.d] = [[[u.name[0], 0], LI]]
    return level_dict


#
#   PURPOSE:    Maps the source, destination vertex names to instances
#   INPUT:      g as an adjacency list. source, destination as string names of the vertices
#   POST-CONDITION: calls _print_path to actually find the shortest parth
def shortest_path(g, source, dest):
    s = g.dict_vertices[source]
    v = g.dict_vertices[dest]
    _print_path(s, v)


#   PURPOSE: Find the shortest path from source to dest, if it exists, and print it
#   INPUT:  s: source vertex instance, v: dest vertex instance
#   POST-CONDITION: prints the shortest path as vertex names, from s to v, if one exists; else print message
def _print_path(s, v):
    if v.name == s.name:
        print s.name
    elif v.pred is None:
        print ("no path from {} to {} exists".format(s, v))
    else:
        _print_path(s, v.pred)
        print v.name

#   PURPOSE: print nicely formatted adjacency list
#   PARAMETERS: list of all vertex objects in graph
#   POST-CONDITION: console displays adjacency list
def printGraph(vertices):
    for vertex in vertices:
        print("%s: [" % vertex.name),
        neighbor_count = 0
        for neighbor in vertex.neighbors:
            neighbor_count += 1
            if neighbor_count < len(vertex.neighbors):
                print("%s,") % neighbor,
            else:
                print neighbor,
        print "]"
    return

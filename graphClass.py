# graphClass.py
# Author:   Jasmine Falk & Andy Valenti
# Source:   https://gist.github.com/anirudhjayaraman/1f74eb656c3dd85ff440fb9f9267f70a
# Date:     8 August 2018

# The graph class method contains two class definitions to allow 
# the user to create a graph and add vertices within the graph.

'''
REVISION HISTORY
1NOV18      Fixed set_freq to set Spanish words to lowercase to be found in the dictionary
19OCT18     Added private variable to Vertex class to store frequency value
            Added function to Vertex class to set the frequency value of a vertex
28AUG18     Modified vertex class variables needed for BFS. Added a dictionary with vertex names as keys as
            vertex instances as values.
18AUG18     Added print functions
13AUG18     Updated documentation
10AUG18     Added documentation
9AUG18      Debugged adjacencyMatrix() method
            Added new methods to Graph class (get_vertex(), store_vertices())
'''

import numpy as np
import os
import pickle
#   ************************************************************************************************
#   *                                         VERTEX CLASS                                         * 
#   ************************************************************************************************

# The Vertex class contains the definition of a Vertex 
# (an object with two fields--a name as a string, and a list of neighboring vertices), 
# and function definitions for adding neighbors to a vertex.
class Vertex:
    def __init__(self, vertex, language='en'):
        self.name = vertex      # UTF-8 code
        self.lang = language
        self.freq = 0           # frequency of word; use for resting activation in RHM (account for frequency effect)
        self.neighbors = []     # neighboring vertices type vertex class
        self.d = 1000000        # distance from source,s; used by BFS(G, s)
        self.pred = None        # predecessor
        self.color = 0          # 0: white, 1: grey, 2: black

    # PURPOSE: add a single neighbor to a given vertex
    # PARAMS:  Vertex object for the neighboring vertex of any given Vertex
    # RETURNS: bool (false if param is not an instance of the Vertex class)
    def add_neighbor(self, neighbor):
        if isinstance(neighbor, Vertex):
            if neighbor.name not in self.neighbors:
                self.neighbors.append(neighbor.name)
                neighbor.neighbors.append(self.name)
                self.neighbors = sorted(self.neighbors)
                neighbor.neighbors = sorted(neighbor.neighbors)
        else:
            return False

    # PURPOSE: add a list of neighbors to a given vertex
    # PARAMS:  Vertex object for the neighboring vertex of any given Vertex
    # RETURNS: bool (false if param is not an instance of the Vertex class)
    def add_neighbors(self, neighbors):
        for neighbor in neighbors:
            if isinstance(neighbor, Vertex):
                if neighbor.name not in self.neighbors:
                    self.neighbors.append(neighbor.name)
                    neighbor.neighbors.append(self.name)
                    self.neighbors = sorted(self.neighbors)
                    neighbor.neighbors = sorted(neighbor.neighbors)
            else:
                return False

    # PURPOSE: set frequency value for the vertex
    # PARAMS:  Vertex object for the neighboring vertex of any given Vertex
    # POST-CONDITION: freq variable updated 
    def set_freq(self, freq_dict):
        vertex_name = self.name[0].lower()
        if vertex_name not in freq_dict:
            print "warning: {}: Unable to find frequency for {}. Set freq to 0.0".format('set_freq', vertex_name.encode('utf-8'))
            self.freq = 0.0
            return
        else:
            self.freq = freq_dict[vertex_name]
            return

        
    def __repr__(self):
        return str(self.neighbors)

#   ************************************************************************************************
#   *                                          GRAPH CLASS                                         *
#   ************************************************************************************************

# The Graph class contains the definition of a Graph (object with a dictionary of Vertex objects), 
# and function definitions for getting, setting, and adding vertices,
# adding edges, and creating an adjacency list and matrix.
class Graph:
    def __init__(self):
        # TODO explore whether self.dict_vertices could replace self.vertices
        self.vertices = {}          # dictionary {vertex_name: [vertex.neighbors]}
        self.dict_vertices = {}     # dictionary {vertex_name: vertex instance}

    # PURPOSE: store all Vertex objects of a graph into a list 
    # PARAMS:  list of concepts (strings)
    # RETURNS: list of concept Vertices
    def store_vertices(self, concepts):
        vertices = []
        for concept in concepts:
            vertex = Vertex(concept)
            vertices.append(vertex)
            self.dict_vertices[concept] = vertex
        return vertices

    # PURPOSE: return the Vertex object corresponding to the given concept name
    # PARAMS:  vertex name (string), list of Vertex objects
    # RETURNS: Vertex object based on the vertex_name if found; else False
    def get_vertex(self, vertex_name, vertices):
        for vertex in vertices:
            if vertex.name == vertex_name:
                return vertex
        print ('ERROR_{}: Unable to find vertex obj for {}'.format('get_vertex', vertex_name)) 
        return False

    # PURPOSE: add a Vertex to the graph
    # PARAMS:  Vertex object to be added
    # RETURNS: none
    def add_vertex(self, vertex):
        if isinstance(vertex, Vertex):
            self.vertices[vertex.name] = vertex.neighbors

    # PURPOSE: add multiple vertices to the graph
    # PARAMS:  list of Vertex objects to be added
    # RETURNS: none
    def add_vertices(self, vertices):
        for vertex in vertices:
            if isinstance(vertex, Vertex):
                self.vertices[vertex.name] = vertex.neighbors

    # PURPOSE: add a new edge to the graph between two existing vertices
    # PARAMS: vertex that edge starts from, vertex that edge goes to
    # RETURNS: none
    def add_edge(self, vertex_from, vertex_to):
        if isinstance(vertex_from, Vertex) and isinstance(vertex_to, Vertex):
            vertex_from.add_neighbor(vertex_to)
            if isinstance(vertex_from, Vertex) and isinstance(vertex_to, Vertex):
                self.vertices[vertex_from.name] = vertex_from.neighbors
                self.vertices[vertex_to.name] = vertex_to.neighbors

    # PURPOSE: add a new edges to the graph using add_edge()
    # PARAMS: list of edges to be added
    # RETURNS: none
    def add_edges(self, edges):
        for edge in edges:
            self.add_edge(edge[0],edge[1])          

    # PURPOSE: create adjacency list for the graph to display edges between vertices
    # PARAMS: none
    # RETURNS: adjacency list corresponding to the graph
    def adjacencyList(self):
        if len(self.vertices) >= 1:
            return [str(key) + ": " + str(self.vertices[key]) for key in self.vertices.keys()]  
        else:
            return dict()
        
    # PURPOSE: create adjacency matrix forg the graph to display edges between vertices
    # PARAMS: none
    # POST-CONDITION adjacency matrix corresponding to the graph, dictionary mapping of int labels to lemma strings
    def adjacencyMatrix(self):
        if len(self.vertices) >= 1:
            self.vertex_names = sorted(self.vertices.keys())
            self.vertex_indices = dict(zip(self.vertex_names, range(len(self.vertex_names))))
            self.vertex_labels = dict(zip(range(len(self.vertex_names)),self.vertex_names))
            self.adjacency_matrix = np.zeros(shape=(len(self.vertices),len(self.vertices)))
            for i in range(len(self.vertex_names)):
                for j in range(i, len(self.vertices)):
                    for el in self.vertices[self.vertex_names[i]]:
                        j = self.vertex_indices[el]
                        self.adjacency_matrix[i,j] = 1
            return self.adjacency_matrix, self.vertex_labels
        else:
            return dict(), dict()

    # PURPOSE: prints graph as an adjacency list
    # PARAMS: Graph object
    # RETURNS: adjacency list and matrix
def printList(g):
    adjacencyList = g.adjacencyList()
    for vertex in adjacencyList:
        print vertex
    return

# PURPOSE: print graph as adjacency matrix
# PARAMS: Graph object
# RETURNS: none
def printMatrix(g):
    print str(g.adjacencyMatrix()[0])
    return

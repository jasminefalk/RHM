# bilingual_graph.py
# Author:   Jasmine Falk & Andy Valenti
# Date:     9 August 2018

# This script takes a list of English concepts entered by the user and uses semantic vectors 
# to build a graph that organizes these words by their coactivations and Spanish translations
# to represent how bilinguals activate words in a bilingual lexicon.

'''
REVISION HISTORY
02NOV18     TEMPORARY--changed translation provider from Microsoft to MyMemory due to being blocked from using
            Microsoft Azure. TODO: MyMemory isn't as good of a translation package, need to determine how to
            get Microsoft Azure back
01NOV18     Changed loadFreq() to be able to store frequency values for Spanish words also
26OCT18     Unpickle dictionary of frequency values. For each vertex in the graph, loadFreq() finds the freq
            value corresponding to the word and sets the vertex attribute (freq) using the vertex class set_freq
            method to store the frequency value
04OCT18     Fixed cognates overwriting each other's language attribute in self.dict_vertices and self.vertices
            by ensuring all English strings are lower case, utf-8 and all Spanish strings are upper case utf-8
            Note: adj list .csv and adj matrix .graphml files now written to separate dirs adj_list, adj_matrix
02OCT18     Added helper function so that each neighbor in adj list is paired with its language
29SEP18     Fixed translate code to replace Google with Microsoft (free trial 2M characters/mo)
21SEP18     Changed translate package to (https://pypi.org/project/translate/) install: pip install translate
27AUG18     Store language of lemma in vertex.lang; write lang to .csv file
25AUG18     Added external graph visualization capability by exporting g to graphml format
23AUG18     Added method to write the adjacency list to a CSV file; use argparse for command line options
20AUG18     Added capability to toggle between languages in addToGraph() method
19AUG18     Added recursion for Spanish tree
            Changed translate package to googletrans
18AUG18     Added function to print graph
17AUG18     Automatically translates words (removed hardcoded words)
16AUG18     Added capability to build corresponding Spanish tree
            Cleaned up function definitions (fewer params, etc.)
            Updated documentation
15AUG18     Incorporated new kNN method, updateConcepts()
            Added translation Vertex for each concept
10AUG18     Updated documentation
'''
__author__ = 'Jasmine Falk & Andy Valenti'
__copyright__ = "Copyright 2018. Tufts University"

#   Run on command line:
#   python bilingual_graph.py [-g] -b {int} -d {int} -e file_name -s file_name
#
#   include optional parameter -g graph adjacency matrix
#
#   ************************************************************************************************
#   *                                        SET PARAMETERS                                        *
#   ************************************************************************************************

import sys, datetime
import csv
import os
import argparse

#   Text translation is used to obtain the L2 labels given the L1 and vice versa
#   There are three known providers: Google, Microsoft, and MyMemory
#   Google, provides very good tranlations but now regulates and blocks free api calls
#   Microsoft's service is available on a free trial basis from Azure's Cognitive Services. Limit 2M chars/mo.
#   MyMemory is a crowd-sourced translation service. Although free, it works on segments; translating individual words
#   is not its intended purpose
# from googletrans import Translator
from translate import Translator        # python alternative interface to MyMemory (default) or Microsoft (via Azure)

from graphClass import Vertex, Graph, printMatrix
from kNN import loadVectors, loadVectors_unicode, kNN, updateConcepts

# for plotting adjacency matrix
import matplotlib.pyplot as plt
import networkx as nx
import pickle

# Key to authorize Microsoft Azure Cognitive Service Acess to translate text API
#   Note limit is 2m characters per month
secret = "50b20ac12b2c41e58cc389877ae4f929"

# parse command line parameters
parser = argparse.ArgumentParser(description='generate bilingual memory graph')
parser.add_argument('-g','--graph', help='graph adjacency matrix', action='store_true')
parser.add_argument('-d', '--depth', action='store', dest='evoc_depth', type=int, help='depth of evocations')
parser.add_argument('-b', '--breadth', action='store', dest='evoc_breadth', type=int, help='number of evocations per word')
parser.add_argument('-e', '--english', action='store', dest='eng_vectors', type=str,
                    help='file name of english semantic vectors')
parser.add_argument('-s', '--spanish', action='store', dest='span_vectors', type=str,
                    help='file name of spanish semantic vectors')
options = parser.parse_args()
GRAPH = True if options.graph else False

semantic_vector_dir = "../Semantic_vectors/"
# semantic_vector_dir = "/home/jasminefalk/bilingual_mem/Semantic_vectors/"                   # CHANGE THIS BACK BEFORE PUTTING IN BILINGUAL_MEM
eng_vectors = semantic_vector_dir + options.eng_vectors
span_vectors = semantic_vector_dir + options.span_vectors

k = options.evoc_breadth        # Number of nearest neighbors per concept
d = options.evoc_depth          # Depth/height of graph (# of evocation levels)

time_stamp = str(datetime.datetime.now().strftime('%Y-%m-%dT%Hh%Mm%S'))

MIN_DIST = 0.0

#   ************************************************************************************************
#   *                                      INITIALIZE GRAPH                                        *
#   ************************************************************************************************

#   Simple interface to read in concepts
print(" \n*** Bilingual memory graph generator ***")

concepts_input = raw_input('\nEnter 1 or more concepts separated by a comma:').lower().split(',')
concepts_input = [unicode(str.strip(c),'utf-8') for c in concepts_input]   # remove any trailing/leading spaces

if concepts_input[0] == '':
    concepts_input = [unicode('rage','utf-8'), unicode('injure','utf-8'), unicode('lunch','utf-8'), 
                      unicode('religious','utf-8'), unicode('wrong','utf-8'), 
                      unicode('benefit','utf-8'), unicode('smell','utf-8')]

# Convert user inputs to list of tuples
concepts_en = []
for concept in concepts_input:
    concept_tuple = (concept, MIN_DIST)
    concepts_en.append(concept_tuple)

concepts_sp = []  # Spanish translations get added as tree gets built recursively

print(" \n*** Generator parameters ***")
print("Concepts: \t{}".format(concepts_en))
print("Depth of evocations: \t\t%d" % d)
print('Breadth of evocations: \t\t%d' % k)
print('English semantic vector loc:\t%s' % eng_vectors)
print('Spanish semantic vector loc: \t%s' % span_vectors)
if GRAPH:
    print('Graphing adjacency matrix')

# Build data structures to store semantic vectors globally for determining kNN
# Note: all vectors are converted to "utf-8". L1 vectors are converted to lower case
#       L2 vectors are converted to upper case
# Note: must create another list of just concept names (no tuples)
def removeTuples(tuples_list):
    no_tuples = []
    for el in tuples_list:
        no_tuples.append(el[0])
    return no_tuples

evoked_en, evoking_en = loadVectors_unicode(removeTuples(concepts_en), eng_vectors,L1=True)     # English lexicon
evoked_sp, evoking_sp = loadVectors_unicode(concepts_sp, span_vectors,L1=False)    # Spanish lexicon

# Initialize graph
g = Graph()

# Create vertices from the concepts store in running list of the all existing Vertices in the graph
all_vertices = g.store_vertices(concepts_en)
g.add_vertices(all_vertices)

#   ************************************************************************************************
#   *                                  FUNCTIONS TO BUILD GRAPH                                    *
#   ************************************************************************************************

#   PURPOSE: Translate a given English word to its Spanish equivalent
#   PARAMETERS: Concept in English (string)
#   RETURNS: Concept in Spanish (utf-8 format, upper case)
def en2sp(concept):
    # for non-Google translate
    # translator = Translator(provider='microsoft', from_lang='en', to_lang='es', secret_access_key=secret)
    translator = Translator(from_lang='en', to_lang='es')                       # removed microsoft for now
    translation = translator.translate(concept).replace(" ", "_")
    if type(translation) is str:    # translator occasionally returns type str
        translation = unicode(translation,"utf-8")
    return translation.upper()


#   PURPOSE: Translate a given Spanish word to its English equivalent
#   PARAMETERS: Concept in Spanish (string)
#   RETURNS: Concept in English (utf-8 format, lower case)
def sp2en(concept):
    # for new non-Google translate
    # translator = Translator(provider='microsoft', from_lang='es', to_lang='en', secret_access_key=secret)
    translator = Translator(from_lang='es', to_lang='en')
    translation = translator.translate(concept).replace(" ", "_")
    if type(translation) is str:    # translator occasionally returns type str
        translation = unicode(translation,"utf-8")
    return translation.lower()


#   PURPOSE: Add translation of concept to graph as Vertex object
#   PARAMETERS: concept in L2, Vertex object of concept in L1
#   RETURNS: L2 concept Vertex 
def addTranslation(L2_concept, L1_vertex, lang):
    # Convert translation from string to Vertex object
    L2_vertex = Vertex(L2_concept, lang)
    L1_vertex.add_neighbor(L2_vertex)
    g.add_vertex(L2_vertex)
    all_vertices.append(L2_vertex)
    g.dict_vertices[L2_concept] = L2_vertex
    return L2_vertex


#   PURPOSE: Add evocation vertices to graph
#            Update the adjacency lists and matrix to include evocations
#   PARAMETERS: concept tuple (vertex object), list of evocations, language
#   RETURNS: none
def addEvocations(concept_vertex, evocations, lang):
    # List of vertex objects (convert evocations (i.e. nearest neighbors) from strings to Vertices)
    neighbors = []
    for evocation in evocations:
        vertex = Vertex(evocation, lang)
        neighbors.append(vertex)
        g.dict_vertices[evocation] = vertex

    # Add evocation Vertices to graph
    concept_vertex.add_neighbors(neighbors)
    g.add_vertices(neighbors)
    # Update concept_vertex in graph to include all evocation Vertices
    g.add_vertex(concept_vertex)

    # Update list of all existing vertices for next recursive iteration
    for neighbor in neighbors:
        all_vertices.append(neighbor)
    return 


#   PURPOSE: recursive function to create the graph   
#   PARAMETERS: concept (tuple of name & dist), depth of tree, language (defaut set to English)
#   RETURNS: none
def addToGraph(L1_concept, depth, language='en'):
    lang_set = {"en", "sp"}
    L2_name = min(lang_set - {language})  # get the L2 name

    L1_vertex = g.get_vertex(L1_concept, all_vertices)
    if not isinstance(L1_vertex, Vertex):
        sys.exit("Fatal Error: instance is not a vertex")

    # Translate L1 concept to get L2 concept and add to L2 evoking list
    if language == 'en':
        L2_concept = (en2sp(L1_concept[0]), L1_concept[1])
        updateConcepts(L2_concept, evoked_sp, evoking_sp)
    else:
        L2_concept = (sp2en(L1_concept[0]), L1_concept[1])
        updateConcepts(L2_concept, evoked_en, evoking_en)

    # BASE CASE (no more levels to be added)
    if depth == 0:
        if language == 'en':
            L2_concept = (en2sp(L1_concept[0]), L1_concept[1])
        else:
            L2_concept = (sp2en(L1_concept[0]), L1_concept[1])
        addTranslation(L2_concept, L1_vertex, L2_name)
        return
    else:
        if language == 'en':
            # List of tuples of (name, distance_to_parent_concept)
            L1_evocations = kNN(L1_concept[0], evoked_en, evoking_en, k)
            L2_evocations = kNN(L2_concept[0], evoked_sp, evoking_sp, k)
        else:
            L1_evocations = kNN(L1_concept[0], evoked_sp, evoking_sp, k)
            L2_evocations = kNN(L2_concept[0], evoked_en, evoking_en, k)

        # Add L1 and L2 list of most similar words to graph
        addEvocations(L1_vertex, L1_evocations, language)
        L2_vertex = addTranslation(L2_concept, L1_vertex, L2_name)
        addEvocations(L2_vertex, L2_evocations, L2_name)

        for evocation in L1_evocations:
            # Add evocations of given concept to concept_set
            if language == 'en':
                updateConcepts(evocation, evoked_en, evoking_en)
                addToGraph(evocation, depth-1, 'en')
            else: 
                updateConcepts(evocation, evoked_sp, evoking_sp)
                addToGraph(evocation, depth-1, 'sp')

        for evocation in L2_evocations:
            if language == 'en':
                updateConcepts(evocation, evoked_sp, evoking_sp)
                addToGraph(evocation, depth-1, 'sp')
            else:
                updateConcepts(evocation, evoked_en, evoking_en)
                addToGraph(evocation, depth-1, 'en')

#   ************************************************************************************************
#   *                                GRAPH VISUALIZATION FUNCTIONS                                 *
#   ************************************************************************************************

#   PURPOSE: print formatted adjacency list
#   PARAMETERS: list of all vertex objects in graph
#   POST-CONDITION: console displays adjacency list
def printGraph(vertices):
    for vertex in vertices:
        print ("%s: [" % vertex.name[0].encode('utf-8')),
        neighbor_count = 0
        for neighbor in vertex.neighbors:
            neighbor_count += 1
            if neighbor_count < len(vertex.neighbors):
                print "(%s, %.3f)," % (neighbor[0].encode('utf-8'), neighbor[1]),
            else:
                print "(%s, %.3f)" % (neighbor[0].encode('utf-8'), neighbor[1]),
        print "]"
    return


#   PURPOSE: draw a simple visualization (unreadable if graph is dense)
#   PARAMETERS: graph in networkx format
#   POST-CONDITION: plot with verticies arranged as a circle
def draw_adj_matrix(gnx):
    nx.draw_circular(gnx)
    plt.axis('equal')


#   PURPOSE: convert graph object to networkx formation; relabel using mapping
#   PARAMETERS: graph, dictionary in format {int(old_label): str(new_label)}
#   POST-CONDITION: returns a networkx graph with lemma-labeled vertices
def adj_matrix2nx(g, mapping):
    gnx = nx.from_numpy_matrix(g.adjacencyMatrix()[0])
    return nx.relabel_nodes(gnx, mapping)


#   PURPOSE: write graph as an graphml file (XML)
#   PARAMETERS: graph in networkx format, file_name to create
#   POST-CONDITION: file_name.graphml file created from gnx
def write_adj_matrix(gnx, file_name):
    nx.write_graphml_lxml(gnx, file_name)


#   purpose: write adjacency list to csv file
#   input:  name for output file
#   output: each row is a lemma and its evocations written to .csv file
def write_lexicon_csv(filename):

    # purpose: helper function to get language of neighbors
    # input: list of neighbors each a string
    # returns: list in format [neigh: lang, neigh: lang]
    def get_language(neighbor_list):
        name_lang = []
        for neighbor in neighbor_list:
            neighbor_lang = g.dict_vertices[neighbor].lang
            item = "{}: {}".format(neighbor[0].encode('utf-8') ,neighbor_lang)
            name_lang.append(item)
        return name_lang

    max_neighbors = 0

    fout = open(filename, "wb")
    with fout:
        writer = csv.writer(fout, delimiter=',')

        #   find max neighbors and build header
        for vertex in all_vertices:
            if len(vertex.neighbors) > max_neighbors:
                max_neighbors = len(vertex.neighbors)
        header = ["evoc" + str(d) for d in range(1, max_neighbors + 1)]
        header.insert(0,'lemma')
        header.insert(1,'lang')
        writer.writerow(header)

        for vertex in all_vertices:
            row = vertex.neighbors[:]
            new_element = get_language(row)
            # print("try: []".format(new_element))
            new_element.insert(0,vertex.name[0].encode('utf-8'))
            new_element.insert(1,vertex.lang)
            writer.writerow(new_element)


#   purpose save graph object
#   parameters: graph object g, filename to save
def save_graph(g, filename):
    with open(filename, 'wb') as output:
        pickle.dump(g, output, pickle.HIGHEST_PROTOCOL)

#   ************************************************************************************************
#   *                                         BUILD GRAPH                                          *
#   ************************************************************************************************

for concept in concepts_en:
    # print concept
    addToGraph(concept, d)
    print '\n******** %s ********' % concept[0].lower()
    # Print adjacency list to standard output
    printGraph(all_vertices)
    print


#   PURPOSE: use the frequency dictionary to set the freq variable in the Vertex class
#   PARAMETERS: none
#   POST-CONDITION: returns dict object 
def loadFreq():
    print 'Loading frequencies...'

    #   PURPOSE: use pickle module to unserialize dict object to load frequencies
    #   PARAMETERS: none
    #   POST-CONDITION: returns dict object 
    def unpickle_freq(filename):
        if not os.path.isfile(filename):
            print 'Dict file does not exist: {}'.format(filename)

        with open(filename, 'rb') as input:
            # dict_obj contains words and their frequencies
            dict_obj = pickle.load(input)
        return dict_obj

    # Save dictionary of words and their corresponding frequencies
    eng_freq = unpickle_freq("engFreq_dict.pkl")                          # HARDCODED
    span_freq = unpickle_freq("spanFreq_dict.pkl")

    for v in all_vertices:
        if v.lang == 'en':
            v.set_freq(eng_freq)
        else:
            v.set_freq(span_freq)
        print v.name, v.freq

loadFreq()

# Create visualization for graph
write_lexicon_csv('../adj_list/{}_{}_adj_list_{}.csv'.format(d, k, time_stamp))
if GRAPH:
    a, mapping = g.adjacencyMatrix()
    gnx = adj_matrix2nx(g, mapping)
    draw_adj_matrix(gnx)
    write_adj_matrix(gnx, "../adj_matrix/{}_{}_adj_matrix_{}.graphml".format(d, k, time_stamp))
    plt.show()

# Prompt use to save their graph visualization
save_model = raw_input("save model (y/n)?")
if save_model == "y":
    print("saving model")
    graph_objs = {'graph': g, 'concepts_en': concepts_en, 'concepts_sp': concepts_sp, 'all_vertices': all_vertices}
    save_graph(graph_objs, 'graph.pkl')

sys.exit("Remember. Bilingualism is not only fun, it's good for you, too!")

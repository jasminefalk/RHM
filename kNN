# kNN.py
# Author:   Jasmine Falk & Andy Valenti
# Source:   https://machinelearningmastery.com/
#           tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
# Date:     9 August 2018

# The kNN method contains function definitions to calculate the similarity between words based on 
# their semantic vectors. The method reads a csv file of semantic vector data and determines the 
# k most similar neighbors for a given word using Minkowski distance as the measure of similarity.

'''
REVISION HISTORY
10SEP18     Stored vector distances in kNeighbors as name-distance tuples
23AUG18     Implemented Minkowski distance calculation as a replacement for Euclidean distance
20AUG18     Added error message to catch edge cases where there is no vector for a concept
15AUG18     New method, updateConcepts(), to be used to keep track of evocations added to graph
13AUG18     Updated documentation
10AUG18     Separated modules in order to build dataset separately 
            from retrieving similarities to improve algorithm efficiency
            Updated documentation
'''
__author__ = 'Jasmine Falk & Andy Valenti'
__copyright__ = "Copyright 2018. Tufts University"

import sys
import csv
import math
import operator
from decimal import Decimal

#   Initialize constants
FEATURE_THRESHOLD = 50      # number of features below which a vector may be considered 'missing'
P_VALUE = 2.0               # root value for Minkowski distance cal
#   ************************************************************************************************
#   *                                                                                              *
#   *                                     FUNCTION DEFINITIONS                                     *
#   *                                                                                              *
#   ************************************************************************************************

#   ************************************************************************************************
#   *                                           LOAD DATA                                          *
#   ************************************************************************************************


#   PURPOSE: read in data from a given CSV file
#   PARAMETERS: name of a CSV file, L1 is true if English semantic vectors are to be read
#   RETURNS: list of data from the CSV file
#            labels are converted to utf-8 lower case if L1 and utf-8 upper case if L2
def loadDataset_unicode(filename, L1=True):
    # Increase the field size limit to accomodate csv with large fields
    csv.field_size_limit(sys.maxsize)
    data = []
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        dataset = list(lines)
        for row in range(len(dataset)):
            if L1:
                dataset[row][0] = unicode(dataset[row][0], 'utf-8').lower()
            else:
                dataset[row][0] = unicode(dataset[row][0], 'utf-8').upper()
            for col in range(1, len(dataset[row])):
                dataset[row][col] = float(dataset[row][col])
            data.append(dataset[row])
    return data

# TODO this function is deprecated
#   PURPOSE: read in data from a given CSV file
#   PARAMETERS: name of a CSV file
#   RETURNS: list of data from the CSV file
def loadDataset(concepts_set, filename):
    # Increase the field size limit to accomodate csv with large fields
    csv.field_size_limit(sys.maxsize)
    data = []
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        dataset = list(lines)
        for row in range(len(dataset)):
            for col in range(1, len(dataset[row])):
                dataset[row][col] = float(dataset[row][col])
            data.append(dataset[row])
    return data


#   PURPOSE: remove elements of the dataset that are in the concepts_set
#   PARAMETERS: dataset (list of vectors)
#   RETURNS: dataset with concept vectors removed
def removeConcepts(concepts_set, dataset):
    evocations = []
    for row in range(len(dataset)):
        if dataset[row][0] not in concepts_set:
            for col in range(1, len(dataset[row])):
                dataset[row][col] = float(dataset[row][col])
            evocations.append(dataset[row])
    return evocations

#   ************************************************************************************************
#   *                                    CALCULATE SIMILARITIES                                    *
#   ************************************************************************************************

#   PURPOSE: calculate Euclidean distance as similarity measure 
#            between two given data instances
#   PARAMETERS: data instance 1, data instance 2, length of the vector
#   POST-CONDITION: returns Euclidean distance
def euclidean_distance(instance1, instance2, length):
    distance = 0
    for i in range(1, length):
        distance += pow((instance1[i] - instance2[i]), 2)
    return math.sqrt(distance)


#   PURPOSE: calculate the Minkowski distance, a generalized form of Euclidean and Manhattan distance
#            between two given data instances
#   PARAMETERS: two vectors: x, y and p_value: the order or the Minkowski metric
#               p_value = 1 is the Manhattan distance
#                       = 2 is the Euclidean distance
#                       = inf is the Chebyshev distance (rarely used)
#               Aggarwal et al (2001) suggests fractional p's (< 1) work better for high dim (dim > 20) data sets
#               "On the Surprising Behavior of Distance Metrics in High Dimensional Space"
#   POST-CONDITION: returns distance between the two vectors
def p_root(value, n_root):
    root_value = 1/float(n_root)
    return round(Decimal(value) ** Decimal(root_value), 3)


def minkowski_dist(x, y, p_value):
    return p_root(sum(pow(abs(a-b), p_value) for a, b in zip(x[1:], y[1:])), p_value)


#   PURPOSE: collect the k most similar instances for a given unseen instance
#   PARAMETERS: dataset (with no concepts), concept to be tested (full vector list), k value
#   RETURNS: list of k most similar words (neighbors)
def getNeighbors(dataset, testInstance, k):
    distances = []

    length = len(testInstance)
    if length < FEATURE_THRESHOLD:
        print('ERROR: {}: Unable to find semantic vector for {}'.format('getNeighbors', testInstance[0]))    # TODO
        return []

    for i in range(len(dataset)):
        # dist = euclidean_distance(testInstance, dataset[i], length)
        dist = minkowski_dist(testInstance, dataset[i], P_VALUE) # TODO try different fract p_values
        distances.append((dataset[i], dist))
    distances.sort(key=operator.itemgetter(1))

    # Return only the k nearest neighbors (k smallest distances)
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i])
    return neighbors

#   ************************************************************************************************
#   *                                             PRINT                                            *
#   ************************************************************************************************

#   NOTE: not used in graph.py, but available for other uses
#   PURPOSE: print out all the concepts with their k nearest neighbors
#   PARAMETERS: semantic vectors list of evocations, semantic vectors list of concepts, k
#   RETURNS: none
def printAll(evocations, concepts, k):
    for i in range(len(concepts)):
        print concepts[i]
        nearest_neighbors = getNeighbors(evocations, concepts[i], k)
        print '*** %s ***' % concepts[i][0].upper()
        for neighbor in nearest_neighbors:
            print '%s %.5f' % (neighbor[0][0], neighbor[1])
        print
    return


#   PURPOSE: print the k nearest neighbors for one concept
#   PARAMETERS: list of nearest neighbors for one concept
#   RETURNS: none
def print_kNN(nearest_neighbors):
    for neighbor in nearest_neighbors:
        print '%s %.5f' % (neighbor[0][0], neighbor[1])
    return

#   ************************************************************************************************
#   *                                                                                              *
#   *                                        USER INTERFACE                                        *
#   *                                                                                              *
#   ************************************************************************************************


#   TODO this function is deprecated
#   PURPOSE: builds data structures to store semantic vectors read in from the given csv file
#   PARAMETERS: name of csv file containing vector values 
#               (defaults to first command-line argument) 
#   RETURNS: evocations vectors list (training set), concepts vectors list (testing set)
def loadVectors(concepts_set, filename):
    dataset = loadDataset(concepts_set, filename)
    # Create separate data sets for concepts and words activated by the concepts
    evocations = removeConcepts(concepts_set, dataset)
    concepts = []
    for target_word in dataset:
        if target_word[0] in concepts_set:
            concepts.append(target_word)
    return evocations, concepts


#   PURPOSE: builds data structures to store semantic vectors read in from the given csv file
#   PARAMETERS: name of csv file containing vector values
#               (defaults to first command-line argument)
#   RETURNS: evocations vectors list, concepts vectors list
def loadVectors_unicode(concepts_set, filename, L1=True):
    dataset = loadDataset_unicode(filename, L1)
    # Create separate data sets for concepts and words activated by the concepts
    evoked = removeConcepts(concepts_set, dataset)
    evoking = []
    for target_word in dataset:
        if target_word[0] in concepts_set:
            evoking.append(target_word)
    return evoked, evoking


#   PURPOSE: used to keep track of evocations added to graph
#   PARAMETERS: concept tuple (name, distance), evocations vectors list, concepts vectors list
#   POST-CONDITION: removes concept from evocations; adds concept to concept vectors list
def updateConcepts(concept, evocations, concepts):
    found = False
    # Remove new concept from evocations list and append to concept list
    for i in range(len(evocations)):
        utf = evocations[i][0]
        # print("type {} utf {}".format(type(utf), utf.encode('utf-8')))
        if evocations[i][0] == concept[0]:
            new_concept = evocations[i]
            evocations.remove(new_concept)
            concepts.append(new_concept)
            found = True
            break
    if not found:
        print('warning: {}: Unable to find evocations vector for {}'.format('updateConcepts', concept[0].encode('utf-8')))
    return


#   PURPOSE: allows user to execute kNN algorithm in one function call c
#   PARAMETERS: concept name (string), evocations vectors list, concepts vectors list, 
#               number of nearest neighbors to find (defaults to k=3)
#   RETURNS: list of k nearest neighbors as tuples (evocation_name, similarity_value)
def kNN(concept, evocations, concepts, k=3):
    found = False
    # Obtain semantic vector corresponding to the given concept name
    for i in range(len(concepts)):
        if concepts[i][0] == concept:
            concept = concepts[i]
            found = True
            break
    if found:
        nearest_neighbors = getNeighbors(evocations, concept, k)

        # Store nearest neighbors of the given concept to be returned
        # kNeighbors: [(neighbor1_name, dist), (neighbor2_name, dist), ... (neighbork_name, dist)]
        kNeighbors = []
        for neighbor in nearest_neighbors:
            kNeighbors.append((neighbor[0][0], neighbor[1]))
        return kNeighbors
    else:
        print('ERROR: {}: Unable to find semantic vector for {}'.format('kNN', concept[0].encode('utf-8')))
        return []

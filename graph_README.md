# Organizing Bilingual Coactivations
The bilingual_graph program allows the user to organize and store words of a bilingual lexicon in a graph based on their coactivations. These coactivations are determined by measuring the Euclidean distance between semantic vectors of a given lexicon to determine their nearest neighbors.
###### A visual of a bilingual graph for the concept "cat".
![Imgur](https://i.imgur.com/8Jr05gI.jpg)

## Getting Started
### Prerequisites
The user must have numPy and googletrans installed to run bilingual_graph. googletrans requires Internet connection.
```
pip install numpy
pip install googletrans
```
### Semantic vector files
The bilingual_graph program requires that the user load in two CSV files of semantic vectors. The words in the first file on the command line are referred to as Lexicon 1, and the second file words make up Lexicon 2. Lexicon 1 is set to English as default and Lexicon 2 to Spanish, but one or both languages can be changed depending on the needs of the user. The files should not contain vectors for punctuation markings or special characters (words with accent markings are fine).
### Parameters
The user can set the following parameters in the bilingual_graph.py code:
* k: number of coactivations the program will determine for each concept
* d: depth or height of the graph
* concept list: list of vertices from which all other coactivations stem from

## Usage
### Running the program
bilingual_graph takes two command line arguments, the name of the file of Lexicon 1 vectors and the name of the file of Lexicon 2 vectors, respectively. 
For example:
```
python bilingual_graph.py Semantic_vectors_English.csv Semantic_vectors_Spanish.csv
```
### Output
The program currently prints out an adjacency list of the graph that it constructs for each concept in the user-entered concept list. 
TODO: print adjacency matrix after mapping matrix to concept names.

## Program modules
bilingual_graph uses two independent modules: kNN and graphClass
### kNN
The kNN method calculates the similarity between words based on their semantic vectors. The method reads a csv file of semantic vector data and determines the k most similar neighbors for a given word using Euclidean distance as the measure of similarity. The kNN method contains three public functions for a user to import:
* **loadVectors():** builds data structures to store the semantic vectors of the concepts and the vectors of the words evoked by the concepts
* **updateConcepts():** as words are added to the graph, remove these concept from list of evoked concepts and append to list of evoking concepts
* **kNN():** calculates the Euclidean distances for a given concept and its evocations, and returns a list of its k nearest neighbors
### graphClass
The graphClass method contains definitions of two classes, a Vertex class and a Graph class, which allows the user of the module to create a graph of Vertex objects. A Vertex object has two fields: the name of the vertex as a string, and a list of its neighboring vertices. The Vertex class has functions that allow the user to add neighbors to a Vertex object. An instance of the Graph class is an object with a dictionary of Vertex objects. The class also contains function definitions for getting, setting, and adding vertices, adding edges, and creating an adjacency list and matrix.

## Control Flow
#### Initialize graph
1. Load semantic vector data from files provided by user for both lexicons. Store vectors as list of evoking concepts and evoked concepts for both languages. 
   ```python
   evoked_en, evoking_en = loadVectors(concepts_en, sys.argv[1])     # English lexicon
   evoked_sp, evoking_sp = loadVectors(concepts_sp, sys.argv[2])     # Spanish lexicon
   ```
2. Initiate instance of Graph class.
   ```python
   g = Graph()
   ```
3. Add concepts evoking concepts to Graph g as Vertex objects.
#### Build graph
```python
for concept in concepts_en:
    addToGraph(concept, depth)
```
4. Add each of the concepts, their translations, their evocations, and their translation's evocations to the graph as Vertex objects with the **addToGraph() method**, which:
      - takes the name of a concept (as a string), the height of the tree (set by the user as d), and the language of the concept (defaults to English).
      - translates the given concept to the language of the other lexicon using the googletrans Python library.
      - determines the k nearest neighbors for both the given concept and its translation using the kNN module.
        ```python
        L1_evocations = kNN(L1_concept, evoked_en, evoking_en, k)
        L2_evocations = kNN(L2_concept, evoked_sp, evoking_sp, k)
        ```
      - adds the k nearest neighbors as Vertex objects to the graph for both L1 and L2.
      - repeats the same process recursively for each of the evocations in the list of the k most nearest neighbors, until d = 0 (base case), which indicates that there are no more levels of evocations to be added to the graph.
        ```python
        for evocation in L1_evocations:
            addToGraph(evocation, depth-1, 'en')
        
        for evocation in L2_evocations:
            addToGraph(evocation, depth-1, 'sp')
        ```
#### Print graph
5. Print an adjacency list to stdout. For instance, for the graph of the concept "cat" for k=3 and d=1 would be printed as:
   ```
   ******** CAT ********
   cat: [ mouse, pet, fur, gato ]
   mouse: [ cat, raton ] 
   pet: [ cat, mascota ]
   fur: [ cat, pelo ]
   gato: [ perro, bigotes, pata, cat ]
   perro: [ gato, dog ]
   bigotes: [ gato, whiskers ]
   pata: [ gato, paw ]
   ```
   *(note: pictured above would be d=2)*
6. TODO: print adjacency matrix that is mapped to concept names.
7. TODO: method to traverse generate graph using breadth-first search to traverse the generated graph for printing and debugging purposes.

## Authors
###### Human-Robot Interaction Lab, Tufts University
* **Jasmine Falk**
* **Andy Valenti**

## Acknowledgments 
* Anirudh Jayaraman ([Python Implementation of Undirected Graph](https://gist.github.com/anirudhjayaraman/1f74eb656c3dd85ff440fb9f9267f70a))
* Jason Brownlee ([Tutorial To Implement k-Nearest Neighbors in Python From Scratch](https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/))

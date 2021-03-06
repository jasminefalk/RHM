# Revised Hierarchical Model
TODO:
* brief overview of architecture of RHM
* diagram of RHM

## Getting Started
### Prerequisites
In order to load coactivations into the model, the user must first have run **bilingual_graph.py** to generate a **graph.pkl** file (see **ORGANIZING BILINGUAL COACTIVATIONS** below for usage details).

### Parameters
The model uses a number of pre-set parameters that can be changed by the user via the SetParams command (TODO: add back into user menu interface).
TODO: list the params here

### Usage
RHM can be run using the following command:
```
python RHM.py
```
A command menu is immediately displayed.
```
******************************************************
      Bilingual Memory Access (RHM)
      Andrew Valenti, Jasmine Falk
      HRI Lab, Tufts University

              A: Auto-load co-activations
              C: Continue cycle
              Display activations:
                  D:    Enter 1 or more lemmas or concepts
                  D1:   All lemmas, singe plot
                  D10:  Top 10 word activations
              Enter Input
                  N:   Enter concept
              Print concepts/lemmas in lexicon
                  PC: Print concepts
                  PL: Print lemmas
              R:  Reset model
              F:  Load frequencies
              X:  Exit program

          Trial: 1 Cycle: 0 Mode: Ready
          
******************************************************
             Please enter an action: 
```
#### Loading coactivations
User types "a" or "A" into the command action line. The user will be prompted to enter the filename of the coactivation graph, which should simply be **graph.pkl**, which was generated by running **bilingual_graph.py**. 

**TODO: Finish documenting other commands**


# Organizing Bilingual Coactivations
The bilingual_graph program allows the user to organize and store words of a bilingual lexicon in a graph based on their coactivations. These coactivations are determined by measuring the Euclidean distance between semantic vectors of a given lexicon to determine their nearest neighbors.

## Getting Started
### Prerequisites
The user must have numPy, networkx, matplotlib, and translate installed to run bilingual_graph.
```
sudo pip install numpy
sudo pip install networkx
sudo pip install matplotlib
sudo pip install translate
```
To visualize the output, the user must have downloaded Cytoscape, an open source platform for visualizing networks (http://www.cytoscape.org/).

### Semantic vector files
The bilingual_graph program requires that the user load in two CSV files of semantic vectors. The words in the first file on the command line are referred to as Lexicon 1, and the second file words make up Lexicon 2. Lexicon 1 is set to English as default and Lexicon 2 to Spanish, but one or both languages can be changed depending on the needs of the user. The files should not contain vectors for punctuation markings or special characters (words with accent markings are fine).

### Parameters
The user can set the following parameters:
* b: breadth of graph (number of evocations the program will determine for each lemma)
* d: depth of graph (number of levels)
* concept list: list of vertices from which all other coactivations stem from

## Usage
### Running the program
bilingual_graph takes the following command line flags and arguments:
```
python bilingual_graph.py [--graph] --breadth {int} --depth {int} --english {english_vecs.csv} --spanish {spanish_vecs.csv}
```
The flags can also be shortened to:
```
python bilingual_graph.py [-g] -b {int} -d {int} -e {english_vecs.csv} -s {spanish_vecs.csv}
```
-g is an optional flag. The user should include it if creating a visualization of the graph adjacency matrix is desired.

As an example, the following command line input causes the program to read in two semantic vector files (called semvec_english.csv and semvec_spanish.csv), create a graph with a depth of 2 levels and with 3 evocations per lemma, and saves the graph to a file that can be used for visualization in Cytoscape.
```
python bilingual_graph.py -g -b 3 -d 2 -e semvec_english.csv -s semvec_spanish.csv
```

### Output
bilingual_graph has the following forms of output:
1. **standard output:** bilingual_graph prints an adjacency list of the graph to standard output so that quick viewing is easily accessible from the terminal 
2. **csv file:** bilingual_graph writes the adjacency list to a csv file that is saved in the same directory that the bilingual_graph.py script is located in. The filename format is as follows:
   ```
   b_d_lex_year-month-dayTtime.csv
   ```
   For instance:
   ```
   2_3_lex_2018-09-14T17h50m23.csv
   ```
3. **graphml file:** bilingual_graph saves the graph in a file of .graphml format to the same directory, which can then be used to visualize the graph interactively in Cytoscape. The filename format is as follows:
   ```
   adj_matrix_year-month-day-Ttime.graphml
   ```
   For instance:
   ```
   adj_matrix_2018-09-14T17h50m23.graphml
   ```
   **NOTE:** a graphml file is only generated when the user includes -g in the command line.
4. **pkl file:** bilingual_graph also saves the graph in pickle file format (.pkl), which can then be loaded into the RHM program.
   **NOTE:** the graph is only saved in .pkl format when the user types "y" when prompted to save their graph.



#### Interpreting the output
1. **standard output:** The following is an example of a hypothetical output that bilingual_graph might print to terminal:
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
The first lemma generated, "cat", is the root of the graph. "cat" evokes 3 similar lemmas, "mouse", "pet", and "fur", and one translation lemma, "gato". Because "mouse" is on the bottom level of this particular graph, it does not evoke any other similar words, and instead, only evokes its Spanish translational equivalent, "raton". "cat" is included in the list for "mouse" because "cat" was the lemma that invoked it.

2. **csv file:** A reference for interpreting the csv file is printed on the first line of each generated csv file.

3. **graphml file:** The graphml file is useful for visualizing the graph in Cytoscape. To see it in Cytoscape, the user should open the Cytoscape application, go to **File > Import > Network > File**, and the choose the correct .graphml file. A graph of the concept "lunch" would be visualized as:
![Imgur](https://i.imgur.com/USilIme.png)

4. **pkl file:** The pickle file is used to load a lexicon into the RHM program (see README_RHM for usage instructions).

## Program modules
bilingual_graph uses two independent modules: kNN and graphClass
### kNN
The kNN method calculates the similarity between words based on their semantic vectors. The method reads a csv file of semantic vector data and determines the k most similar neighbors for a given word using the Minkowski distance as the measure of similarity. The kNN method contains three public functions for a user to import:
* **loadVectors():** builds data structures to store the semantic vectors of the concepts and the vectors of the words evoked by the concepts
* **updateConcepts():** as words are added to the graph, remove these concept from list of evoked concepts and append to list of evoking concepts
* **kNN():** calculates the Minkowski distances for a given concept and its evocations, and returns a list of tuples of its k nearest neighbors and their corresponding distances

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

## Authors
#### Human-Robot Interaction Lab, Tufts University
* **Andy Valenti**
* **Jasmine Falk**

## Acknowledgments 
* Anirudh Jayaraman ([Python Implementation of Undirected Graph](https://gist.github.com/anirudhjayaraman/1f74eb656c3dd85ff440fb9f9267f70a))
* Jason Brownlee ([Tutorial To Implement k-Nearest Neighbors in Python From Scratch](https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/))


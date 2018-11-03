import sys
import math

'''
This data structure implements a model of bilingual memory based on the integrated lexicon of the BIA/BIA+ models.
This connectionist architecture uses the principles of interactive activation (McClelland & Rumelhart, 1982)
Note that there lexicon contains no default entries; the model must be loaded using doAutoLoad() using a previously
generated graph, e.g., graph.pkl
'''

__author__ = 'Andy Valenti and Jasmine Falk'
__copyright__ = "Copyright 2016,2018. Tufts University"

rest = -0.1  # the resting activation level to which activations tend to settle in the absence of external input
alpha = 0.1  # this parameter scales the strength of the excitatory input to units from other units in the network
gamma = 0.1  # this parameter scales the strength of the inhibitory input to units from other units in the network

#Bilingual Lexicon Revised Hierarchical Model Weights
S2E = 1.0   # Spanish to English projection
E2S = 1.0   # English to Spanish projection
L2Ev = 1.0  # (Evoking) Lemma to Evoked word
Ev2L = 1.0  # Evoked word to (Evoking) Lemma
C2L = 1.0   # Concept to Lemma
L2L = 1.0   # Lemma to Lemma
# self-inhibitory weights
LI = -1.0         # Used this for readability during testing/debugging
# LI = -0.21   # lemma to lemma inhibition



k = 10.0  # arbitrary scaling factor in initial computation of the strength or upon model reset



#   Defines an IAC Unit
#   Input: list in the form [[[unit_name,pos] weight], [[unit_name,pos] weight] ... ]
#        unit_name is the name (a string) of the sending unit. Must be a valid key in a dict defining the pool of units
#        pos: indicates the position (or index) of the unit in a grouping of related units. Used in the lets pool to
#             indicate position of letter in a word; otherwise, pos = 0.
#        weight: the positive (excitatory) weight or negative (inhibitory) weight of the projection.
#   Unit vales accessible via get and set methods:
#   Projection list exactly in the form as in Input. Caller can test for projections via isProjNone() method.
#   Activation, net input, and external input for THIS unit
class Unit:
    def __init__(self, projections=None, activation=None):
        if projections is None:
            self.projections = None
        else:
            self.projections = projections

        if activation is None:  # only for the default lexicon
            self.rest = rest
        else:  # when auto loading, rest act is given
            self.rest = activation

        self.activation = self.rest
        self.ext_input = 0.0
        self.net_input = 0.0
        self.running_avg = self.rest
        self.strength = math.exp(k * self.running_avg)

    def isProjNone(self):
        if self.projections is None:
            return True

    def getNumProj(self):
        return len(self.projections)

    def getNetInput(self):
        return self.net_input

    def setNetInput(self, net_input):
        self.net_input = net_input
        return

    def getWeight(self, proj_index):
        if proj_index > len(self.projections):
            print("Error(getWeight): Out of Index")
            sys.exit(1)
        return self.projections[proj_index][1]

    def getProjList(self):
        return self.projections

    def setProjList(self, proj_list):
        self.projections = proj_list
        return

    def getProj(self, proj_index):
        if proj_index > len(self.projections):
            print("Error(getProj): Out of Index")
            sys.exit(1)
        return self.projections[proj_index][0]

    def getExtInput(self):
        return self.ext_input

    def setExtInput(self, ei):
        self.ext_input = ei
        return

    def getActivation(self):
        return self.activation

    def getRunningAvg(self):
        return self.running_avg

    def getStrength(self):
        return self.strength

    def setWeight(self, proj_index, weight):
        if proj_index > len(self.projections):
            print("Error: Projections Out of Index")
            sys.exit(1)
        self.projections[proj_index][1] = weight
        return

    def setActivation(self, activation):
        self.activation = activation
        return

    def setRunningAvg(self, running_avg):
        self.running_avg = running_avg
        return

    def setStrength(self, strength):
        self.strength = strength
        return

    def resetActivation(self):
        self.activation = self.rest
        self.running_avg = self.rest
        self.strength = math.exp(k * self.running_avg)
        return

    def getRest(self):
        return self.rest

    def setRest(self, new_rest):
        self.rest = new_rest
        return


#   ************************************** Pool Architecture, RHM model: **********************************
#   Pools consist of 'Units' which represent artificial neurons
#   There are 3 pools: concepts, words_eng, words_span
#   Words are limited to 15 letters per word.
#   Each unit is initialized with a list containing zero or more 'projection' sublists, one for each connection
#   from a sending unit in a pool. Projections may be from any pool, including itself
#   Each projection sublist contains another sublist with the sending unit's key followed by the position in the
#   corresponding dictionary entry for the key. At present, this is only used for the 'lets' pool (in BIA model only).
#   Projections from other pools have a 0 position.
#   The [key,position] pair is followed by the weight of the projection; this completes a sublist projection.
#
#  ****************************************** CONCEPTS *****************************************************

concepts = {}   # concepts are activated by supplying external input to its unit, either from the console or script (TBD)



#  ******************************************* LEMMAS ******************************************************
#   The LEMMA pool is represented as a graph in which the first level are the concepts, the second level are the lemmas
#   evoked by the concept, the third level are the lemmas invoked by the lemmas at the first level, and so on.
#
#   Self-inhibitory connections for English words and their evocations
#   We theorize that it is sufficient to self-inhibit only those other lemmas at the same depth in the graph that forms
#   the lexicon.
lemmas_level = {}   # dictionary of lemmas per level {level: [lemma1, lemma2, .. lemma_n]}
lemmas = {}         # contains both english and spanish words

# list of all other lemmas at a given level. Each item is in lemma_si_template format
lemma_si_template = [['lemma',0], LI]
lemma_si = []                                           #

# list of all the evoked lemmas that project to an evoking lemma. Each item is in lemma_proj_template format
lemma_proj_template = [['lemma', 0], L2L]
lemma_projections = []


# Using the magic encoding

import csv
import math
import operator
import matplotlib.font_manager

import os, sys, re, copy

import matplotlib.pyplot as plt
from numpy import interp
from lexicon_pools import lemmas, concepts, Unit, L2L, C2L
from itertools import cycle
from helper_class import shortest_path, BFS, build_level_dict
import pickle

'''
REVISION HISTORY
26OCT18     Changed loadFreq() and updateRestAct() methods to be able to map log-frequency values stored 
            in the graph that is loaded in through doAutoload() to resting activations
            Resting activation range changed to MIN_REST_ACT = -0.05 and MAX_REST_ACT = 0.0 based on the range 
            noted in McClelland and Rumelhart's IA model
12OCT18     Distances from concept tuple mapped to C2L and L2L weights
31AUG18     Initial version of loading co-activations from a graph implemented
27AUG18     Began autoload co-activations implementation: load a .pkl (pickle) file to instantiate graph 'g'
31MAY18     Added PC Print Concept command and cleaned-up error input error message 
14APR18     Began constructing code base
'''
#   This is an implementation of Kroll and Stewart's Revised Hierarchical Model of Bilingual Memory
#   Processing is based on McClelland and Rumelhart's  Interactive Activation model as described in
#   Explorations in Parallel Distributed Processing, McClelland, J.L. 1987

__author__ = 'Andy Valenti and Jasmine Falk'
__copyright__ = "Copyright 2018. Tufts University"

plt.rcParams["font.family"] = "DejaVu Sans"

#   Global variables

MAX_WORD_LEN = 15 # Maximum word length
e = 0.0002      # epsilon value. During script processing we stop cycling when activations change by this amount.
ncycles = 0    # number of cycles. It is used to hold value from script.
item_script = ''       # used by runCycle to cycle until item settles within e
list_script = []       # used to process a list of items to display; build from the script
script_mode = False
blankCycle = False      # used to recognize when word is a blank

# model variables
verbose = False
# verbose = True
input_concept = ''
console_message = ''
mode = 'Ready'
cycleno = 0  # current cycle; it is reset by reset()
display_cols = 3.0  # Number of columns in a subplot of the word activations. Needs to be adjusted as words are added
trial = 1  # counts the number of times the model has been reset/initialized. It is never reset by the model.
act_dataset_lemma = []  # accumulates word activations for english
act_langset = []  # accumulates language node activationslet
act_dataset_concepts = []  # accumulates concept node activations
strength_dataset = []  # accumulate word strengths
strength_letterset = []  # accumulate letter strengths
exit_flag = False
log = []
logging = False


pool_list = [lemmas, concepts]
reverse_pool = {'lemmas': lemmas, 'concepts': concepts}
a2z = 'abcdefghijklmnopqrstuvwxyz'
max_subplots = 30   # The most subplots that can be reasonably displayed simultaneously

#   Parameter definitions (from Explorations in Parallel Distributed Processing: A Handbook of Models,
#   Programs, and Exercises. James McClelland. July 28th, 2015)

MAX_ACT= 1.0        # maximum activation parameter
MIN_ACT= -0.2       # minimum activation parameter
MAX_REST_ACT = 0.0   # maximum resting activation parameter (for loadfreq)
MIN_REST_ACT = -0.05 # minimum resting activation parameter 
REST = -0.1         # the resting activation level to which activations tend to settle in the absence of external input
# DECAY = 0.1       # the decay rate parameter, which determines the strength of the tendency to return to resting level
DECAY = 0.07        # McCllelan Default Value. the decay rate parameter, which determines the strength of the tendency to return to resting level
ESTR = 0.4          # this parameter stands for the strength of external input (inputs from outside the network). It scales
                    # the influence of external signals relative to internally generated inputs to units
ALPHA = 0.1        # this parameter scales the strength of the excitatory input to units from other units in the network
GAMMA = 0.1         # this parameter scales the strength of the inhibitory input to units from other units in the network
ncycles_default = 5   # default number of cycles
ORATE = 0.05    # rate of accumulation of activation for the purposes of determining response strength
# oscale corresponds to the parameter k in the equation to compute the response strength
oscale_lets = 10.0  # as recommended in McClelland and Rumelhart, Explorations in Parallel Distributed Processing
oscale_words = 20.0 # as above
params = {'max': MAX_ACT, 'min': MIN_ACT, 'rest': REST, 'decay': DECAY, 'estr': ESTR, 'alpha': ALPHA, 'gamma': GAMMA,
          'ncycles': ncycles_default, 'orate': ORATE}


#  Function Definitions

#   builds a list of keys for a pool, used for charting axis
def build_keys(pool):
    keys = []
    for key in pool:
        keys.append(key)
    list.sort(keys)
    return keys


#   Creates the x values for a dataset of size dataset_size, where each record in the set corresponds
#   to r update cycles of the mode
def genx_axis(r, dataset_size):
    return [x * r for x in range(1, dataset_size)]


# complement of a list of integers
# courtesy J. Eunice, stackoverflow, 2015
def complement(l, universe=None):
    """
    Return the complement of a list of integers, as compared to
    a given "universe" set. If no universe is specified,
    consider the universe to be all integers between
    the minimum and maximum values of the given list.
    """
    if universe is not None:
        universe = set(universe)
    else:
        universe = set(range(min(l), max(l) + 1))
    return sorted(universe - set(l))


# Reset completely restarts the model.
# Need to add code to reset each unit to default activation, rest, net-input, and ext_input values
# NOTES: Does NOT reset the params

def reset_pool(pool):
    for key, unit_list in pool.iteritems():
        for unit in unit_list:
            unit.resetActivation()
            unit.setExtInput(0.0)
            unit.setNetInput(0.0)
    return


def reset():
    global trial, act_dataset_lemma, act_dataset_concepts, verbose, cycleno, console_message, mode,\
        input_concept, params, log, blankCycle, script_mode

    # reset each unit to default activation, rest, net-input, and ext_input values
    reset_pool(concepts)
    reset_pool(lemmas)
    # reset_pool(schemas)
    trial += 1
    cycleno = 0
    verbose = False
    act_dataset_lemma = []
    act_dataset_concepts = []
    act_schemaset = []
    console_message = ''
    mode = 'Ready'
    input_concept = ''
    blankCycle = False
    script_mode = False
    log = []
    return


#  Cycle(act_dataset_*) cycles through the pools, collecting the net input of each unit and then updating the unit
#  activation
#  Input: act_dataset_* is a list which contains a record of a pool's activation for each update across cycles.
#  Control variables: ncycles controls the number of iterations of the net input & update cycle
#                     verbose controls printing of each update cycle to the standard output (console)
#  Returns: act_dataset_lemma, act_dataset_span, act_dataset_concepts appended with the last ncycles activation records
#  for each unit in the pool
def cycle_pool():
    global verbose, cycleno, act_dataset_lemma, act_dataset_concepts
    act_trial_lemma = []

    for reps in range(int(params['ncycles'])):  # ensure ncycles is type int bc doSetParams converts it to float
        cycleno += 1

        # gather netInput from pools
        netInput(concepts)
        netInput(lemmas)
        # netInput(schemas)

        # update the pools
        update(concepts)
        update(lemmas)

        # update(schemas)
        act_dataset_lemma.append(readActivations(lemmas))
        act_dataset_concepts.append(readActivations(concepts))
        act_trial_lemma.append(readActivations(lemmas))

        if verbose is True:
            print('Lemma Activations:')
            print('Cycleno: ' + repr(reps + 1) + ' ' + repr(act_trial_lemma[reps]))

    return act_dataset_lemma, act_dataset_concepts


def extract_col(col_num, dataset):
    col = []
    for cycle_row in dataset:
        col.append(cycle_row[col_num][1])
    return col


#   NEW readActivation using generic pool structure
#   Reads all units in pool and if the pool is legal (a dict obj), returns:
#   act_list: [[unit_name0, activation],..,[unit_name,activation]]
#   Input: a pool such as lemmas, lets, lang and an activation data set.
#          It is up to the caller to initialize an empty dataset.
def readActivations(pool):
    def getKey(item):
        return item[0]
    act_list = []
    for key, unit_list in pool.iteritems():
        posnum = 0
        for unit in unit_list:
            activation = unit.getActivation()
            act_list.append([key + repr(posnum),activation])
            posnum += 1
    act_list = sorted(act_list, key=getKey)
    return act_list


#   function netInput(rcvr_pool) parses the projections in rcvr_pool and looks up activation of each sending unit
#   The standard netInput routine computes the net input for
#   each pool. The net input consists of three things: the external input, scaled by
#   estr; the excitatory input from other units, scaled by alpha; and the inhibitory
#   input from other units, scaled by gamma. For each pool, the netInput routine first
#   accumulates the excitatory and inhibitory inputs from other units, then scales
#   the inputs and adds them to the scaled external input to obtain the net input.
def netInput(rcvr_pool):
    global pool_list, params
    # generic pool function
    for key, unit_list in rcvr_pool.iteritems():
        for unit in unit_list:
            in_pool = False
            excitation = 0
            inhibition = 0
            if not unit.isProjNone():
                for sender in unit.getProjList():
                    #print repr(unit.getProjList())
                    from_keypos = sender[0]
                    from_key = from_keypos[0]
                    from_pos = from_keypos[1]
                    weight = sender[1]
                # check to see if key is in any sending pool, i.e. lets, lemmas, lang
                    for pool in pool_list:
                        # print 'From Key:' + repr(from_key)
                        if from_key in pool:
                            activation = pool[from_key][from_pos].getActivation()
                            in_pool = True
                            break
                        else:
                            in_pool = False
                    if in_pool is False:
                        print('from_key', from_key)
                        print("NetInput: Unrecoverable Error. No pool found.")
                        sys.exit(1)

                    if activation > 0:     # process only positive activations
                        if weight > 0:
                            excitation += weight * activation
                        elif weight < 0:
                            inhibition += weight * activation
            excitation *= params['alpha']
            inhibition *= params['gamma']

            unit.setNetInput(excitation + inhibition + unit.getExtInput()*params['estr'])
            # if rcvr_pool == lemmas_eng:
            #     print('key: %s netInput %.4f ' % (key, excitation + inhibition + unit.getExtInput()*params['estr']))
    return


# Standard update. The update routine computes the activation of each unit,
# based on the net input and the existing activation value.
# NEW: computes the running average of the activation
def update(pool):
    global params
    # generic pool update
    k = oscale_words
    for key, unit_list in pool.iteritems():
        for unit in unit_list:
            activation = unit.getActivation()
            net_input = unit.getNetInput()
            resting_level = unit.getRest()
            if activation > 1.0:
                pool_name = [p_key for p_key, p_value in reverse_pool.iteritems() if p_value == pool][0]
                print('Runaway activation in %s, key: %s: %.4f' % (pool_name, key, activation))

            if net_input > 0:
                activation_change = (params['max'] - activation) * net_input\
                                    - params['decay'] * (activation - resting_level)
                if (activation + activation_change) > params['max']:
                    unit.setActivation(params['max'])
                else:
                    unit.setActivation(activation + activation_change)
            else:
                activation_change = (activation - params['min']) * net_input\
                                    - params['decay'] * (activation - resting_level)
                if (activation + activation_change) < params['min']:
                    unit.setActivation(params['min'])
                else:
                    unit.setActivation(activation + activation_change)
            running_avg = params['orate'] * unit.getActivation() + (1 - params['orate']) * activation
            try:
                s = math.exp(k * running_avg)
            except OverflowError:
                s = float('inf')
            unit.setRunningAvg(running_avg)
            unit.setStrength(s)
    return

#   ***********************************Model's User Interface *******************************************

#   function definitions

#  An input word is a concept. it will find the concept in English and Spanish and return the respective activations
def buildConsoleMsg():
    global console_message
    console_message = 'Input concept: {0:s}'.format(input_concept.upper())
    return

#   Perform an update cycle, so long as a word has been inputted
def doContinue():
    global mode, input_concept, console_message, mode
    if input_concept == '':
        mode = 'Error'
        console_message = 'Unable to cycle until word is entered.'
        return
    mode = 'Cycle'
    cycle_pool()
    buildConsoleMsg()
    return

#   **********************************  DISPLAY ACTIVATIONS ***********************************************************

#   Get and return activation of input concept
def getConceptActivation():
    global input_concept, act_dataset
    input_concept_key = input_concept + '_c'
    if input_concept_key not in concepts:
        concept_act = REST
    else:
        unit = concepts[input_concept_key][0]
        concept_act = unit.getActivation()
    return concept_act


#   [D1] display a single plot with activations, 10 words at-a-time
def doDisplay(dataset):
    global cycleno, console_message, mode, input_concept

    def build_key_list(activation_set):
        key_list = []
        for word_tuple in activation_set:
            # key_list.append(word_tuple[0][:-1].decode('utf-8'))
            key_list.append(word_tuple[0][:-1])
        return key_list

    def build_next_act_list(act_list, i_pos, j_pos):
        next_act_list = []
        sliceObj = slice(i_pos, j_pos)
        for time_step in act_list:
            next_act_list.append(time_step[sliceObj])
        return next_act_list

    action_set = {'p', 'n', 'q'}
    lines = ['--', '-.', ':', '-']
    linecycler = cycle(lines)
    step = 10 # this is number of activations we will display

    if cycleno == 0:
        mode = 'Error'
        console_message = 'No data available until a cycle is run.'
        return
    # key_list_test = build_keys(lemmas)

    buildConsoleMsg()
    plt.xlabel('Cycles')
    plt.ylabel('Activation')
    # fig, ax = plt.subplots()
    # Since we want to be able to handle large lexicon, let the user cycle through 10 at a time
    iterations = divmod(len(dataset[0]), step)  # (quotient, remainder)
    pos = 0
    i_pos = 0

    while True:
        next_act_dataset = build_next_act_list(dataset, i_pos, i_pos + step)
        key_list = build_key_list(next_act_dataset[0])   # TODO fix key_list when utf-8 chars present
        mode = '1-plot, {0:d}-words'.format(len(key_list))
        x_vals = genx_axis(1,len(next_act_dataset)+1)
        for col in range(len(next_act_dataset[1])):
            plt.plot(x_vals, extract_col(col, next_act_dataset), next(linecycler))
        concept_act = getConceptActivation()
        plt.title('Input word: {0:s} Activ: {1:4f}'.format(input_concept.upper(), concept_act))
        plt.legend(key_list, loc='best', ncol=3)
        plt.show()

        action_input = raw_input("Enter [N]ext, [P]rev, [Q]uit: ").lower()
        if action_input not in action_set:
            mode = 'Error'
            console_message = '{:s} not valid action'.format(action_input.upper())
            plt.close()
            return
        elif action_input == 'n':
            if pos < iterations[0]:
                pos += 1
                i_pos += step
            plt.gcf().clear()
        elif action_input == 'p':
            i_pos -= step
            pos -= 1
            if i_pos < 0:
                i_pos = 0
                pos = 0
            plt.gcf().clear()
        elif action_input == 'q':
            plt.close()
            return


#   Display English words 1 plot
def doDisplayEng():
    global act_dataset_lemma
    doDisplay(act_dataset_lemma)
    return

#   easter egg that displays available fonts
def doDisplayFonts():
    font_list = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    font_list.sort()
    for font in font_list:
        print font
    return

#   display Top 10 Activations
def doDisplayTop10():
    rows = 10
    left_margin = 10
    max_word_len = MAX_WORD_LEN + 1
    global act_dataset_lemma, input_concept, display_cols, mode, console_message
    if cycleno == 0:
        mode = 'Error'
        console_message = 'No data available until a cycle is run.'
        return
    mode = 'Top 10 Activs'
    buildConsoleMsg()
    activations_t = list(act_dataset_lemma[-1])  # get latest activations
    activations_t.sort(key=operator.itemgetter(1), reverse=True)

    print ''
    print (5 * ' '),
    print('*** Top 10 Activations ***\n')
    print ((left_margin) * ' '),
    print('Lemma  activations')
    print ((left_margin) * ' '),
    print('----------------')
    for i in range(rows):
        word = activations_t[i][0][:-1]
        word_str = word + (max_word_len - len(word)) * ' '
        print (left_margin * ' '),
        print('%s%+.4f' % (word_str.encode('utf-8'), activations_t[i][1]))

    #print activations_lets_t
    return


#   Display activation of one or more words or concepts entered at input prompt
#   To display concepts, user must enter 'CONCEPT_c'
#   Note: Must have run at least 1 cycle
def doDisplayItems():
    global cycleno, console_message, mode, act_dataset_lemma, input_concept, script_mode, list_script
    lines = ['--', '-.', ':', '-']
    linecycler = cycle(lines)
    # must have processed at least one update cycle
    if cycleno == 0:
        mode = 'Error'
        console_message = 'No data available until a cycle is run.'
        return
    #
    plt.close()
    mode = '1-plot, n-items'
    buildConsoleMsg()
    word_key_list_eng = build_keys(lemmas)
    concepts_key_list = build_keys(concepts)
    key_list = []
    if not script_mode:
        select_item = raw_input('Enter 1 or more items separated by a comma:').split(',')
    else:
        select_item = list_script
    select_item = [unicode(str.strip(c), 'utf-8') for c in select_item]  # remove any trailing/leading spaces
    plt.xlabel('Cycles')
    plt.ylabel('Activation')
    x_vals = genx_axis(1,len(act_dataset_lemma)+1)

    for item in select_item:
        # item = item.lower()
        if item in lemmas:
            item_index = word_key_list_eng.index(item)
            plt.plot(x_vals, extract_col(item_index,act_dataset_lemma),next(linecycler))
            key_list.append(item)
       
        elif item  in concepts:
            item_index = concepts_key_list.index(item)
            plt.plot(x_vals, extract_col(item_index,act_dataset_concepts),next(linecycler))
            key_list.append(item)
        else:
            mode = 'Error'
            console_message = item  + ' Item not found in pool.'
            return
    concept_act = getConceptActivation()
    plt.title('Input concept: {0:s} Activ: {1:4f}'.format(input_concept.upper(),concept_act))
    plt.legend(key_list, loc='best')
    plt.show()
    return


#   Handle user entering a character not found in "case" dictionary
def errhandler():
    global console_message, mode
    mode = 'Error'
    console_message = "Unrecognized action. Try again"
    return


#   print all the word pools, nicely formatted
def printWords_Full(word_pool):
    print('Word Pool')
    for key, value in word_pool.iteritems():
        proj_list = value[0].getProjList()
        word_len = len(key)
        num_words = len(proj_list)
        print_rows = int(math.floor(num_words / word_len))
        rem_cols = num_words % word_len
        print('\n' + key.upper() + ' projs: ' + repr(num_words))

        # print rows with no. cols. = word length
        for i in range(0,print_rows):
            j = i * word_len
            if word_len == 3:
                print ('{0:s},{1:s},{2:s}'.format(proj_list[j],proj_list[j+1],proj_list[j+2]))
            elif word_len == 4:
                print ('{0:s},{1:s},{2:s},{3:s}'.format(proj_list[j],proj_list[j+1],proj_list[j+2],proj_list[j+3]))
            elif word_len == MAX_WORD_LEN:
                print ('{0:s},{1:s},{2:s},{3:s},{4:s}'.format(proj_list[j],proj_list[j+1],proj_list[j+2],
                                                              proj_list[j+3],proj_list[j+4]))
            else: break

        # here we print the remaining words which fill a partial row
        j = (i + 1) * word_len
        if rem_cols == 1:
            print ('{0:s}'.format(proj_list[j]))
        elif rem_cols == 2:
            print ('{0:s},{1:s}'.format(proj_list[j],proj_list[j+1]))
        elif rem_cols == 3:
            print ('{0:s},{1:s},{2:s}'.format(proj_list[j],proj_list[j+1],proj_list[j+2]))
        elif rem_cols == 4:
            print ('{0:s},{1:s},{2:s},{3:s}'.format(proj_list[j],proj_list[j+1],proj_list[j+2],proj_list[j+3]))
    return


def printWords(word_pool):
    cols = 6    # max columns
    separator = 13 # space between words; should be len of longest word
    global mode
    mode = 'Print lemmas/concepts'
    buildConsoleMsg()

    lexicon = build_keys(word_pool)
    all_rows = divmod(len(lexicon), cols)
    full_rows = all_rows[0]
    partial_rows = all_rows[1]

    row_idx = -1        # handles the case when there is only one partial row
    if word_pool != concepts:
        print('Lemmas:')
        for row_idx in range(full_rows):
            for col_idx in range(cols):
                print lexicon[col_idx + row_idx * cols].encode('utf-8') + (
                        separator - len(lexicon[col_idx + row_idx * cols])) * ' ',
            print('\n')
        row_idx += 1
        if partial_rows != 0:
            for col_idx in range(partial_rows):
                print lexicon[col_idx + row_idx * cols].encode('utf-8') + (
                        separator - len(lexicon[col_idx + row_idx * cols])) * ' ',
            print('\n')
    else: # for concepts, we have to trim the suffix
        print('Concepts:')
        for row_idx in range(full_rows):
            for col_idx in range(cols):
                print repr(lexicon[col_idx + row_idx * cols][:-2]) + (
                        separator - len(lexicon[col_idx + row_idx * cols])) * ' ',
            print('\n')
        row_idx += 1
        if partial_rows != 0:
            for col_idx in range(partial_rows):
                print repr(lexicon[col_idx + row_idx * cols][:-2]) + (
                        separator - len(lexicon[col_idx + row_idx * cols])) * ' ',
            print('\n')


def printLemmas():
    printWords(lemmas)
    return

def printConcepts():
    printWords(concepts)
    return


#   Process a new input concept from console.
#   Set ext input to the concept in the pool
def doNewConcept():
    global input_concept, trial, console_message, mode, script_mode, item_script, blankCycle
    # clear the ext_input from the last word
    mode = 'New Concept'
    blankCycle = False
    if len(input_concept) != 0:
        concepts[input_concept + '_c'][0].setExtInput(0)
    if not script_mode:
        input_concept = raw_input('Enter a concept: ')
    else:
        input_concept = item_script

    # input_concept = input_concept.lower()
    input_concept_key = input_concept.lower() + '_c'

    if input_concept_key not in concepts:
        console_message = 'Concept not found (Enter PC for choices): {0:s}'.format(input_concept.upper())
        input_concept = ''
        return
    else:
        concepts[input_concept_key][0].setExtInput(1)
    buildConsoleMsg()
    return

#   Turns verbose flag on/off
def doLogging():
    global verbose, mode, console_message, input_concept
    console_message = 'You entered: {0:s}'.format(input_concept)
    mode = 'Verbose'
    if verbose is True:
        verbose = False
    else: verbose = True
    return


def doExit():
    print 'Adios!'
    sys.exit(0)


#   *********************************** External File Processing ******************************************************
# CSV Reader: used for reading a script and for auto-loading a new word
# open  CSV file in universal new line mode to handle Unicode encoding.
def load_csv(file_str):
    i = 0
    csv_pattern = re.compile(',')
    script = []
    success = False
    if os.path.exists(file_str):
        with open(file_str,'rU') as f:
            script_reader = csv.reader(f, dialect=csv.excel_tab)
            for i, row in enumerate(script_reader):
                split_row = csv_pattern.split(row[0])
                # print split_row
                script.append(split_row)
        print('*** Read (%d) words ***' % (i + 1))
        success = True
    else:
        print "*** File {} was not found ***".format(file_str)
    return script, success


#   write frequency list to CSV file
def writeFreq(file_str, log):
    with open('norm_' + file_str, 'wb') as csvfile:
        logwriter = csv.writer(csvfile)
        for rec in log:
            logwriter.writerow(rec)
    csvfile.close()
    return


'''
  Read co-activation graph from file and build concept, word pools
  Steps to build lexicon.
  Concepts pool:
    (1) For every concept in concepts_en not in concepts pool:
            add 'concept_c': [Unit()] to concepts{}. 
             
            neighbors_proj.extend(word_si_list)
            add words['lemma_eng'] = [Unit(neighbors_proj, resting_activation)]
'''
def doAutoLoad():
    global mode, console_message, a2z, rest

    # PURPOSE: normalize a given Euclidean distance value to projection weight values
    # PARAMS: Euclidean distance for a given word
    # RETURNS: normalized weight value
    def dist_to_weight(dist):
        # Lower distances get mapped to higher weights
        weight = interp(dist, [0, 15], [1, 0])
        return weight

    graph_file = raw_input('             Please enter the co-activation graph filename: ')
    if not os.path.isfile(graph_file):
        mode = 'Error: Auto-load'
        console_message = 'Graph file does not exist: {}'.format(graph_file)
        return

    with open(graph_file, 'rb') as input:
        graph_objects = pickle.load(input)
    g = graph_objects['graph']
    all_vertices = graph_objects['all_vertices']
    concepts_en = graph_objects['concepts_en']
    concepts_sp = graph_objects['concepts_sp']

    # test to see if loading the saved graph object works
    print '\n*** Building Pools. This may take a moment ***'
    print ("\tenglish concepts: {}".format(concepts_en))

    # root "concept lemma" processing
    # For every concept in concepts_en not in concepts pool:
    # add 'concept_c': [Unit()] to concepts{}.
    for concept in concepts_en:
        concept_key = concept[0] + '_c'
        if concept_key not in concepts:
            concepts[concept_key] = [Unit()]
            print("\n*** adding concept input unit: {} ***".format(concept_key.encode('utf-8')))

        # build depth-first search tree for concept (assumes concepts are 'roots' of separate graphs)
        nodes_visited = BFS(g, concept, clear=True)

        # builds a dictionary of self-inhibitory connections with level# (distance from concept) as key
        lemma_si_dict = build_level_dict(g)

        # concept level to lemma pool processing
        # build projection list of all lemmas evoked by 'concept lemma' (lemma supplied with external input from
        # concept unit) plus the concept input unit # L2L_weight = dist_to_weight(neighbor[1])

        C2L_weight = dist_to_weight(concept[1])
        projections = [[[concept_key, 0], C2L]]  # projections template
        projections = [[[concept_key, 0], C2L_weight]]    # projections template

        for neighbor in g.dict_vertices[concept].neighbors: 
            L2L_weight = dist_to_weight(neighbor[1])
            projections.append([[neighbor[0], 0], L2L_weight])

        # the inhibitory connections includes all the other 'concept lemmas'.
        # Note: we assume each concept forms a forest of sub-graphs (trees). If not, processing might have to change.
        concept_si_list = []

        # si_concepts = concepts_en[:]    # copy so as not to mutate original list
        si_concepts = [c[0] for c in concepts_en]  # copy list (remove tuples) so as not to mutate original list
        si_concepts.remove(concept[0])                # remove the concept in focus, leaving other concepts

        for si_concept in si_concepts: # L2L_weight = dist_to_weight(neighbor[1])
            concept_si_list.append([[si_concept + '_c', 0], -C2L_weight])

        projections.extend(concept_si_list)
        lemmas[concept[0]] = [Unit(projections)]   # TODO add resting_activation when available from word freq processing
        print('\n*** adding projections to concept lemma {}'.format(concept[0].encode('utf-8')))
        for p in projections:
            print p[0][0].encode('utf-8'),
            print '%.3f' % p[1]

        # Evoked lemmas processing
        for vertex in nodes_visited:
            if vertex.name[0] not in lemmas:
                projections = []
                for neighbor in vertex.neighbors:
                    L2L_weight = dist_to_weight(neighbor[1])
                    projections.append([[neighbor[0], 0], L2L_weight])
                lemma_si = lemma_si_dict[vertex.d][:]
                # Remove the current lemma from its inhibitory list
                for idx, val in enumerate(lemma_si):
                    if val[0][0] == vertex.name[0]:
                        del lemma_si[idx]
                        break
                projections.extend(lemma_si)
                lemmas[vertex.name[0]] = [Unit(projections)]

                print('\n*** Adding projections to lemma {} ***'.format(vertex.name[0].encode('utf-8')))
                for p in projections:
                    print p[0][0].encode('utf-8'),
                    print '%.3f' % p[1]

    loadFreq(all_vertices)

    mode = 'Auto-loading co-activations'
    console_message = 'Co-activation graph loaded'

    # TODO can this be deleted now??
    resting_activation = params['rest']  # we can alter the default resting level for auto-loaded words this way

    return g, all_vertices, concepts_en, concepts_sp

#   PURPOSE: maps frequencies to resting activations for each vertex in the graph
#            to account of the frequency effect
#   PARAMS: list of all vertices in the loaded graph
#   POST CONDITION: unit rest variable updated with mapped resting activation
def loadFreq(all_vertices):
    global MAX_REST_ACT, MIN_REST_ACT, mode
    mode = 'Auto-loading frequencies'

    # Get frequencies of all Vertices in Graph
    word_freq = {}
    for v in all_vertices:
        word = v.name[0]
        freq_value = v.freq
        word_freq[word] = freq_value

    # Determine range of frequencies for loaded lexicon
    min_freq = min(word_freq.itervalues())
    max_freq = max(word_freq.itervalues())
    freq_range = [min_freq, max_freq]

    #   PURPOSE: normalize a given word log-frequency to a resting activation value
    #   PARAMS: log-frequency value, range of frequencies in loaded lexicon
    #   RETURNS: normalized resting activation value
    def freq_to_restAct(freq, freq_range):
        # Vertex with word that has the highest frequency gets MAX_REST_ACT
        rest_act = interp(freq, freq_range, [MIN_REST_ACT, MAX_REST_ACT])
        return rest_act

    # Normalize all frequency values to resting activation values
    for word in word_freq:
        freq = word_freq[word]
        rest_act = freq_to_restAct(freq, freq_range)
        # Replace frequency value in dictionary with rest_act value
        word_freq[word] = rest_act

    # Use Unit class function to set the rest class variable for all lemmas
    updateRestAct(word_freq)
    return

#   PURPOSE: update the resting activation of every unit in the lemmas pool
#   PARAMS: dictionary of word-activation pairs
#   POST-CONDITION: rest variable for every unit in lemmas pool updated with mapped resting activation
def updateRestAct(act_dict):
    global lemmas

    for word in lemmas:
        lemma_unit = lemmas[word]
        lemma_unit[0].setRest(act_dict[word])
    
    # FOR TESTING
    print
    for word in lemmas:
        lemma_unit = lemmas[word]
        print word, lemma_unit[0].getRest()
    return


#   *********************************** Character-based Menu Interface ************************************************

def showBanner():
    print
    print '******************************************************'
    print '      Bilingual Memory Access (RHM)'
    print '      Andrew Valenti, Jasmine Falk'
    print '      HRI Lab, Tufts University'
    print
    print '              A: Auto-load co-activations'
    # print '              B: Blank word cycle'
    print '              C: Continue cycle'
    # print '              Cue activation:'
    # print '                  C1:  L1 cue activation'
    # print '                  C2:  L2 cue activation'
    print '              Display activations:'
    print '                  D:    Enter 1 or more lemmas or concepts'
    print '                  D1:   All lemmas, singe plot'
    # print '                  D1S:  All Spanish lemmas, singe plot'
    # print '                  D2:  Subplot lemmas'
    print '                  D10:  Top 10 word activations'
    print '              Enter Input'
    print '                  N:   Enter concept'
    # print '              P:  Set model parameters'
    print '              Print concepts/lemmas in lexicon'
    print '                  PC: Print concepts'
    print '                  PL: Print lemmas'
    # print '                  PS: Print Spanish lexicon'
    # print '              Response Probability:'
    # print '                  RL: Letter choice'
    # print '                  RW: Word choice'
    print '              R:  Reset model'
    # print '              F:  Load frequencies'
    print '              X:  Exit program'
    print
    print ('          Trial: %d Cycle: %d Mode: %s' % (trial, cycleno, mode))
    print ('          %s' % console_message)
    print '******************************************************'
    return

#   User interface is implemented as a switch using a Python dictionary
takeaction = {
    'A': doAutoLoad,
    # 'B': doBlankCycle,
    'C': doContinue,
    'D': doDisplayItems,
    'D1': doDisplayEng,
    # 'D2': doDisplaySubPlots,
    'D10': doDisplayTop10,
    # 'L': doDisplayLang,
    # 'P': doSetParams,
    # 'PAZ': printLets,
    'DF': doDisplayFonts,
    'PC': printConcepts,
    'PL': printLemmas,
    # 'C1': doSetCue1,
    # 'C2': doSetCue2,
    'N':  doNewConcept,
    'R': reset,
    # 'RW': wordChoice,
    'F': loadFreq,
    # 'T': doLogging,
    'X': doExit
}
#   ****************************Model's User Interface Processing Loop*********************************
plt.ion()   # turn on interactive charting
while True:
    showBanner()
    action = raw_input('             Please enter an action: ')
    takeaction.get(action.upper(),errhandler)()
    # plt.show()

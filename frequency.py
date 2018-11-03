# frequency.py
# Author:   Jasmine Falk
# Date:     19 October 2018

'''
REVISION HISTORY
28OCT18     
'''

__author__ = 'Jasmine Falk'
__copyright__ = "Copyright 2018. Tufts University"

import sys
import math
import csv
import pickle

eng_freq_file = "../word_freq/english_logfreq.csv"
span_freq_file = "../word_freq/spanish_lemmas20k.csv"

eng_file_save = "engFreq_dict.pkl"
span_file_save = "spanFreq_dict.pkl"

#   PURPOSE: read in data from a given CSV file
#   PARAMETERS: name of a CSV file with word freq values
#   RETURNS: list of tuples of (word, freq_value)
#            labels are converted to utf-8 lower case
def load_freq(filename):
    # Increase the field size limit to accomodate csv with large fields
    csv.field_size_limit(sys.maxsize)
    data = []
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        dataset = list(lines)
        for row in range(1,len(dataset)):           # Range starts at 1 to ignore header
            dataset[row][0] = unicode(dataset[row][0], 'utf-8').lower()
            data.append(list(dataset[row]))
    return data

#   PURPOSE: convert frequency values to their corresponding log frequencies
#   PARAMETERS: list of tuples of (word, freq_value)
#   RETURNS: list of tuples of (word, log_freq)
def to_logFreq(data):
    for pair in data:
        freq_value = float(pair[1])
        log_freq = math.log(freq_value)
        pair[1] = log_freq
    return data


#   PURPOSE: store list of tuples as dict with key=word, value=freq
#   PARAMETERS: list of tuples of (word, freq_value)
#   RETURNS: dict of word names as keys and frequencies as values
def load_dict(data):
    freq_dict = {}
    for pair in data:
        word = pair[0]
        freq_value = float(pair[1])
        freq_dict[word] = freq_value 
    return freq_dict


#   PURPOSE: save list object to disk using pickle
#   PARAMETERS: frequency list object, filename to save to
#   POST-CONDITION: list of frequencies saved to disk
def save_dict(freq_dict, file_save):
    with open(file_save, 'wb') as output:
        pickle.dump(freq_dict, output, pickle.HIGHEST_PROTOCOL)


def eng_frequency():
    data = load_freq(eng_freq_file)
    freq_dict = load_dict(data)
    save_dict(freq_dict, eng_file_save)
    print 'English frequencies successfully saved to %s' % eng_file_save


def span_frequency():
    data = load_freq(span_freq_file)
    data = to_logFreq(data)
    freq_dict = load_dict(data)
    save_dict(freq_dict, span_file_save)
    print 'Spanish frequencies successfully saved to %s' % span_file_save


eng_frequency()
span_frequency()

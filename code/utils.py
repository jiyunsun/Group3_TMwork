import csv
import string
from collections import Counter

def read_in_conll_file(conll_file, delimiter='\t'):
    '''
    Read in conll file and return structured object

    :param conll_file: path to conll_file
    :param delimiter: specifies how columns are separated. Tabs are standard in conll

    :returns structured representation of information included in conll file
    '''
    my_conll = open(conll_file, 'r', encoding='utf-8')
    conll_as_csvreader = csv.reader(my_conll, delimiter=delimiter, quoting=csv.QUOTE_NONE)
    return conll_as_csvreader

def list_of_tokens(csv_object):
    tokens = []
    for row in csv_object:
        if len(row) > 0:
            tokens.append(get_token(row))
    return tokens


def get_token(row):
    token =  row[3].lower()
    #token_nopunct = token.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    return token

def get_negcue_label(row):
    return row[4]

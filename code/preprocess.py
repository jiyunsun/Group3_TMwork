import csv
import sys
import nltk
import spacy
import string
from utils import *
from feature_extraction import *



def add_feature_columns(conll_file, outputfilename, lexicon):
    '''
    Check which annotations need to be converted for the output to match and convert them

    :param conll_object: structured object with conll annotations
    :param annotation_identifier: indicator of how to find the annotations in the object (e.g. key of dictionary, index of list)
    '''
    conll_object = read_in_conll_file(conll_file)
    with open(outputfilename, 'w') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        header = ['doc_id', 'sentence_id', 'token_id', 'token', 'pos', 'start_ngrams', 'end_ngrams', 'sbs_count', 'negcue']
        csvwriter.writerow(header)
        for row in conll_object:
            if len(row) <= 0:
                continue
            token = get_token(row)
            if not token:
                continue
            pos = nltk.pos_tag([token])[0][1]
            start_ngrams, end_ngrams, sbs_count = create_ngram_features(token, lexicon, n=5)
            row[3] = token
            row.insert(4, pos)
            row.insert(5, start_ngrams)
            row.insert(6, end_ngrams)
            row.insert(7, sbs_count)

            csvwriter.writerow(row)

def main(args=None):
    if not args:
        args = sys.argv
    path = args[1]
    conll = read_in_conll_file(path)
    tokenlist = list_of_tokens(conll)
    lexicon =  generate_lexicon(tokenlist)
    add_feature_columns(path, r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\SEM-2012-SharedTask-CD-SCO-simple.v2\SEM-2012-SharedTask-CD-SCO-training-simple.v2-preprocessed.txt', lexicon)
    #print(tokenlist)

args = ['x', r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\SEM-2012-SharedTask-CD-SCO-simple.v2\SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt']
main(args)

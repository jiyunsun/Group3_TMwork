import csv
import sys
import nltk
import spacy
import string
from utils import *

def remove_blanks(conll_file, outputfilename):
    conll_object = read_in_conll_file(conll_file)
    with open(outputfilename, 'w', newline='') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        for row in conll_object:
            if len(row) <= 0:
                continue

            csvwriter.writerow(row)

def add_feature_columns(conll_file, outputfilename, lexicon, tokenlist):
    '''
    Preprocess the file by converting the content and adding feature columns.
    Converts:
        -tokens to lowercase
        -labels from BIO-NEG(cues) to binary format (1 if NegCue, 0 otherwise)
    Adds:
        - POS tag
        -

    :param conll_object: structured object with conll annotations
    :param
    '''
    conll_object = read_in_conll_file(conll_file)
    with open(outputfilename, 'w', newline='') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        header = ['doc_id', 'sentence_id', 'token_id', 'token', 'lemma', 'prev', 'next','pos', 'punct', 'suffix', 'infix', 'prev_tok_cue', 'sbs_count', 'match_one', 'match_multi', 'negcuelabel']
        csvwriter.writerow(header)
        prev_token_label = 0
        prev_token = ''
        doc = tokenlist2doc(tokenlist)
        for d in doc:
            print(d, end='-')
        for i, (row, token) in enumerate(zip(conll_object, doc)):
            #print(token, 'xxx', row)
            lemma = get_lemma(token)
            pos = get_PoS(token)
            ends_with_suffix = has_neg_suff(token.text)
            infix = has_negation_infix(token.text)
            punctuation = is_punctuation(token.text)
            try:
                next_token = get_lemma(doc[i + 1])
            except IndexError:
                next_token = ''
            start_ngrams, end_ngrams, sbs_count = create_ngram_features(token.text, lexicon, n=5)
            matches_one = matches_oneword_negexpr(token.text)
            matches_multi = matches_multiword_negexpr(token.text, i, doc)
            #row[3] = token.text
            row.insert(4, lemma)
            row.insert(5, prev_token)
            row.insert(6, next_token)
            row.insert(7, pos)
            row.insert(8, punctuation)
            row.insert(9, ends_with_suffix)
            row.insert(10, infix)
            row.insert(11, prev_token_label)
            #row.insert(6, start_ngrams)
            #row.insert(7, end_ngrams)
            row.insert(12, sbs_count)
            row.insert(13, matches_one)
            row.insert(14, matches_multi)

            label = get_negcue_label(row, -1)
            row[-1] = label
            prev_token_label = label
            prev_token = token
            csvwriter.writerow(row)

def main(args=None):
    if not args:
        args = sys.argv
    path = args[1]
    conll = read_in_conll_file(path)
    tokenlist = list_of_tokens(conll)
    lexicon =  generate_lexicon(tokenlist)
    no_blanks_path = r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\SEM-2012-SharedTask-CD-SCO-simple.v2\SEM-2012-SharedTask-CD-SCO-training-simple.v2_rem_nl.txt'
    remove_blanks(path, no_blanks_path)
    out_path = r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\SEM-2012-SharedTask-CD-SCO-simple.v2\SEM-2012-SharedTask-CD-SCO-training-simple.v2-preprocessed.txt'
    add_feature_columns(no_blanks_path, out_path, lexicon, tokenlist)
    #print(tokenlist)

args = ['x', r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\SEM-2012-SharedTask-CD-SCO-simple.v2\SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt']
main(args)

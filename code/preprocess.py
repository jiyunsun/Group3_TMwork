import csv
import sys
import nltk
import spacy
import string
from utils import *

def remove_blanks(conll_file, outputfilename):
    ''' Checks if the file contains any blank lines and removes them'''

    conll_object = read_in_conll_file(conll_file)
    with open(outputfilename, 'w', newline='') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        for row in conll_object:
            if len(row) <= 0:
                continue

            csvwriter.writerow(row)

def add_feature_columns(conll_file, outputfilename, tokenlist):
    '''
    Preprocess the file by converting the content and adding feature columns.
    Converts: labels from BIO-NEG(cues) to binary format (NEG if NegCue, O otherwise)
    Adds:
        - token lemma
        - token POS tag
        - previous token
        - next token
        - is_punctuationmark (or not) as 0 or 1
        - contains negation suffix as 0 or 1
        - contains negation infix as 0 or 1
        - contains negation prefix as 0 or 1
        - if the previous token was a negation cue as 0 or 1
        - the number of occurences of the longest character ngram of the substring if the token contains a negation prefix
        - if the token matches a one-word negation expression as 0 or 1
        - if the token matches a multi-word negation expression as 0 or 1

    :param conll_object: structured object with conll annotations
    :param outputfilename: the path to write the preprocessed file todo
    :param tokenlist: the list of tokens in the corpus (ordered)
    '''
    conll_object = read_in_conll_file(conll_file)


    with open(outputfilename, 'w', newline='') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        header = ['doc_id', 'sentence_id', 'token_id', 'token', 'lemma', 'prev', 'next','pos', 'punct', 'prefix', 'suffix', 'infix', 'prev_tok_cue', 'sbs_count', 'match_one', 'match_multi', 'negcuelabel']
        csvwriter.writerow(header)
        # initialize previous_token information as empty
        prev_token_label = 0
        prev_token = ''
        doc = tokenlist2doc(tokenlist) # create a SpaCy pipeline
        lexicon =  generate_lexicon(doc)
        for i, (row, token) in enumerate(zip(conll_object, doc)):
            lemma = get_lemma(token)
            pos = get_PoS(token)
            prefix = has_neg_prefix(token.text)
            ends_with_suffix = has_neg_suff(token.text)
            infix = has_negation_infix(token.text)
            punctuation = is_punctuation(token.text)
            try:
                next_token = get_lemma(doc[i + 1])
            except IndexError: # we want to make sure the program does not crash if reaching the end of the file
                next_token = ''
            start_ngrams, end_ngrams, sbs_count = create_ngram_features(token.text, token.tag_, lexicon, n=5)
            matches_one = matches_oneword_negexpr(token.text)
            matches_multi = matches_multiword_negexpr(token.text, i, doc)

            label = get_negcue_label(row, -1)
            row[-1] = label # this replaces the BIO label with the binary label
            row = row[:4] + [lemma, prev_token,next_token, pos, punctuation, prefix, ends_with_suffix, infix, prev_token_label, sbs_count, matches_one, matches_multi, label]
            prev_token_label = label # update the previous token to current token
            prev_token = token
            csvwriter.writerow(row)

def main(args=None):
    if not args:
        args = sys.argv
    path = args[1]
    conll = read_in_conll_file(path)
    tokenlist = list_of_tokens(conll)

    no_blanks_path = r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\SEM-2012-SharedTask-CD-SCO-simple.v2\SEM-2012-SharedTask-CD-SCO-training-simple.v2_rem_nl.txt'
    remove_blanks(path, no_blanks_path) # remove the blank lines before further processing
    out_path = r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\SEM-2012-SharedTask-CD-SCO-simple.v2\SEM-2012-SharedTask-CD-SCO-training-simple.v2-preprocessed.txt'
    add_feature_columns(no_blanks_path, out_path, tokenlist)
    #print(tokenlist)

args = ['x', r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\SEM-2012-SharedTask-CD-SCO-simple.v2\SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt']
main(args)

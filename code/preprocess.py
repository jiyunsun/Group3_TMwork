import csv
import sys
import nltk
import spacy
import string
from utils import *

def remove_blanks(conll_file, outputfilename):
    ''' Checks if the file contains any blank lines and removes them
        param: conll_file: the path to the conllfile to preprocess (str)
        param: outputfilename: output path (str)'''

    conll_object = read_in_conll_file(conll_file)

    with open(outputfilename, 'w', newline='') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        for row in conll_object:
            if len(row) <= 0: # Checks if empty row and skips those
                continue

            csvwriter.writerow(row) # rewrite

def add_feature_columns(conll_file, outputfilename, tokenlist):
    '''
    Preprocess the file by converting the content and adding feature columns.
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

    B=0 # COUNTS of classes
    I=0
    O=0
    with open(outputfilename, 'w', newline='') as outputcsv:

        csvwriter = csv.writer(outputcsv, delimiter='\t')
        header = ['doc_id', 'sentence_id', 'token_id', 'token', 'lemma', 'prev', 'next','pos', 'punct', 'prefix', 'suffix', 'infix', 'prev_tok_cue','sbs_count', 'match_one', 'match_multi',  'negcuelabel']
        csvwriter.writerow(header)
        # initialize previous_token information as empty
        prev_token_label = 0
        prev_token = ''
        doc = tokenlist2doc(tokenlist) # create a SpaCy pipeline
        lexicon =  generate_lexicon(doc) # generate the lexicon we need for the sbs_count feature
        for i, (row, doc_token) in enumerate(zip(conll_object, doc)):
            token = doc_token.text.lower()
            lemma = get_lemma(doc_token)
            pos = get_PoS(doc_token)
            prefix = has_neg_prefix(token)
            ends_with_suffix = has_neg_suff(token)
            infix = has_negation_infix(token)
            punctuation = is_punctuation(token)
            try:
                next_token = get_lemma(doc[i + 1])
            except IndexError: # we want to make sure the program does not crash if reaching the end of the file
                next_token = ''
            start_ngrams, end_ngrams, sbs_count = create_ngram_features(token, doc_token.tag_, lexicon, n=5)
            matches_one = matches_oneword_negexpr(token)
            matches_multi = matches_multiword_negexpr(token, i, doc)

            label = get_negcue_label(row, -1)
            if label == 'B-NEG':
                B +=1
            elif label == 'I-NEG':
                I +=1
            else:
                O +=1
            row = row[:3] + [token, lemma, prev_token,next_token, pos, punctuation, prefix, ends_with_suffix, infix, prev_token_label, sbs_count, matches_one, matches_multi, label]
            prev_token_label = label # update the previous token to current token
            prev_token = lemma
            csvwriter.writerow(row) # write everything

    print('Number of B-NEG tags: {}\n number of I-NEG tags: {}\n number of O tags: {}\n'.format(B,I,O))


def main(args=None):
    if not args:
        args = sys.argv
    path = args[1]
    conll = read_in_conll_file(path)
    tokenlist = list_of_tokens(conll)

    no_blanks_path =  path[:-4]+ '-rem_nl.txt'
    remove_blanks(path, no_blanks_path) # remove the blank lines before further processing
    out_path = path[:-4]  + '-preprocessed.txt'
    add_feature_columns(no_blanks_path, out_path, tokenlist)
    #print(tokenlist)


if __name__ == "__main__":
    main()

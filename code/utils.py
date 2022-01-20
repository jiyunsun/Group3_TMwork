import csv
import string
from collections import Counter
from nltk import ngrams
import spacy
from spacy.tokenizer import Tokenizer
import re

def custom_tokenizer(nlp):
    # this code was taken from https://stackoverflow.com/questions/57945902/force-spacy-not-to-parse-punctuation
    prefix_re = re.compile('''^\$[a-zA-Z0-9]''')
    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search)


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

def get_negcue_label(row, id):
    return row[id][2:]

def tokenlist2doc(token_list):
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = custom_tokenizer(nlp)
    str = ' '
    doc = str.join(token_list)
    return nlp(doc)


def get_lemma(spacy_token):
    return spacy_token.lemma_

def get_PoS(spacy_token):
    return spacy_token.tag_


def has_neg_suff(token):
    if token.endswith('less'):
        return 1
    return 0

def has_negation_infix(token):
    #CONTAINS NEGATION INFIX:
    if token.endswith("ness"): #instances like "carelessness"
        cut_token_1 = token[:-4]
        if cut_token_1.endswith("less"):
            return 1
    elif token.endswith("ly"): #instances like "restlessly"
        cut_token_2 = token[:-2]
        if cut_token_2.endswith("less"):
            return 1
    return 0

def matches_oneword_negexpr(token):
    negations = ["no", "not", "nor", "neither", "non", "without", "never", "cannot", "n't", "none", "nothing", "nobody",
                        "nowhere"]
    if token in negations:
        return 1
    return 0

def matches_multiword_negexpr(token, i, doc):
    multi_negations = ["no longer", "by no means", "not for the world", "on the contrary", "rather than",
                       "nothing at all", "no more"]
    three_prev = []
    three_next = []
    for j in range(1, 4):
        try:
            three_prev.append(doc[i+j].text)
        except IndexError:
            three_prev.append("")
        try:
            three_next.append(doc[i-j].text)
        except IndexError:
            three_next.append("")
    #print(three_prev, three_next)
    bigram = token + " " + three_prev[0]#create bi-grams and check if they match a multi_negation
    reverse_bigram = three_next[0] + " " + token
    trigram = three_prev[0] + " " + token + " " + three_next[0]
    left_trigram = three_prev[1]+ " " + three_prev[0] + " " + token
    right_trigram = token + " " + three_next[0] + " " + three_next[1]
    left_fourgram = three_prev[2] + " " +three_prev[1] + " " + three_prev[0] + " " + token
    left_center_fourgram = three_prev[1] + " " + three_prev[1] + " " + token + three_next[0]
    right_center_fourgram = three_prev[0]+ " " + token + " " + three_next[0] + " " + three_next[1]
    right_fourgram =   token + " " + three_next[0] + " " + three_next[1] + " " + three_next[2]

    n_grams = [bigram, reverse_bigram, trigram, left_trigram, right_trigram, left_fourgram, left_center_fourgram, right_center_fourgram, right_fourgram]

    if any(expr in multi_negations for expr in n_grams):
        return 1
    else:
        return 0

def is_punctuation(token):
    if token in string.punctuation:
        return 1
    return 0

def generate_lexicon(tokenlist, n=5):
    lexicon = Counter()
    lexicon.update(tokenlist)
    for token in tokenlist:
        if len(token) < n:
            n_new = len(token)
        else:
            n_new = n
        ngrams = generate_ngrams(token, n_new)[0] # start ngram
        lexicon.update(ngrams)
    return lexicon

def generate_ngrams(token, n):
    start_igrams = []
    end_igrams = []
    #print(token)
    for i in reversed(range(1, n+1)):
        #print(i)
        igrams = [''.join(gram) for gram in ngrams(token, i)]
        start_igrams.append(igrams[0])
        end_igrams.append(igrams[-1])

        #print('test', igrams[0])
    #print(start_igrams, end_igrams)
    return start_igrams, end_igrams

def create_ngram_features(token, lexicon, n=5):
    neg_prefixed = ['un', 'in', 'im', 'il', 'dis', 'ir', 'non']
    affixless_token = ''
    for prefix in neg_prefixed:
        if token.startswith(prefix):
            affixless_token = token[len(prefix):]
    if token.endswith('less'):
        affixless_token = token[:-len('less')]
    if len(affixless_token) > 0:
        if len(affixless_token) < n:
            n = len(affixless_token)
        start_ngrams, end_ngrams = generate_ngrams(affixless_token, n)
        return start_ngrams, end_ngrams, lexicon[start_ngrams[0]] # first (longest start ngram of affixless token is the match we search for)
    if len(token) < n:
        n = len(token)
    start_ngrams, end_ngrams = generate_ngrams(token, n)
    return start_ngrams, end_ngrams, 0

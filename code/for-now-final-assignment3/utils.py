import csv
import string
from collections import Counter
from nltk import ngrams
import spacy
from spacy.tokenizer import Tokenizer
import re

def custom_tokenizer(nlp):
    # this code was taken from https://stackoverflow.com/questions/57945902/force-spacy-not-to-parse-punctuation to match our corpus
    prefix_re = re.compile('''^\$[a-zA-Z0-9]''')
    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search)


def read_in_conll_file(conll_file, delimiter='\t'):
    '''
    THIS CODE WAS ADAPTED FROM THE ML4NLP COURSE
    Read in conll file and return structured object

    :param conll_file: path to conll_file
    :param delimiter: specifies how columns are separated. Tabs are standard in conll

    :returns structured representation of information included in conll file
    '''
    my_conll = open(conll_file, 'r', encoding='utf-8')
    conll_as_csvreader = csv.reader(my_conll, delimiter=delimiter, quoting=csv.QUOTE_NONE)
    return conll_as_csvreader

def list_of_tokens(csv_object):
    ''' This function creates a list of tokens from a csv reader object'''
    tokens = []
    for row in csv_object:
        if len(row) > 0:
            tokens.append(get_token(row))
    return tokens


def get_token(row):
    '''This function converts the token to lowercase'''
    token =  row[3].lower()
    #token_nopunct = token.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    return token

def get_negcue_label(row, id):
    '''This function retrieves the label and removes the BIO-aspect'''
    if row[id] != 'O':
        return 1
    return 0


def tokenlist2doc(token_list):
    '''This function applies the custion SpaCy pipeline to our list of tokens'''
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = custom_tokenizer(nlp)
    str = ' '
    doc = str.join(token_list)
    return nlp(doc)


def get_lemma(spacy_token):
    '''Returns the lemma of the token'''
    return spacy_token.lemma_

def get_PoS(spacy_token):
    ''' Returns the POS of the token'''
    return spacy_token.tag_

def has_neg_prefix(token):
    '''Checks if the token starts with a negation prefix. If this is the case, return
        1, else 0'''
    neg_prefixed = ['un', 'in', 'im', 'il', 'dis', 'ir', 'non']
    if any(token.startswith(prefix) for prefix in neg_prefixed):
        return 1
    return 0

def has_neg_suff(token):
    '''Checks if the token ends with the negation suffix -less. If so, return 1 else 0'''
    if token.endswith('less'):
        return 1
    return 0

def has_negation_infix(token):
    '''Check if the token contains the negation infix -less and return 1 if true, 0 otherwise'''
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
    '''Check if the token matches a one-word negation expression. Return 1 if true, 0 otherwise'''
    negations = ["no", "not", "nor", "neither", "non", "without", "never", "cannot", "n't", "none", "nothing", "nobody",
                        "nowhere"]
    if token in negations:
        return 1
    return 0

def matches_multiword_negexpr(token, i, doc):
    '''Return 1 if the token is part of a multi-word negation expression, 0 otherwise.
        :param token: the token (str)
        :param i: the inded of the current token (int)
        :param doc: the SpaCy processed list of tokens which is ordered (doc[i]=token)'''
    multi_negations = ["no longer", "by no means", "not for the world", "on the contrary", "rather than",
                       "nothing at all", "no more"]
    three_prev = [] # we create containers for the three previous tokens in the corpus
    three_next = [] # and for the three next
    for j in range(1, 4):
        # At the start/end of the corpus if there are no prev/next tokens, we will use empty strings
        try:
            three_prev.append(doc[i+j].text)
        except IndexError:
            three_prev.append("")
        try:
            three_next.append(doc[i-j].text)
        except IndexError:
            three_next.append("")

    # we create all possible bi-, tri- and fourgram for each token
    bigram = token + " " + three_prev[0]
    reverse_bigram = three_next[0] + " " + token
    trigram = three_prev[0] + " " + token + " " + three_next[0]
    left_trigram = three_prev[1]+ " " + three_prev[0] + " " + token
    right_trigram = token + " " + three_next[0] + " " + three_next[1]
    left_fourgram = three_prev[2] + " " +three_prev[1] + " " + three_prev[0] + " " + token
    left_center_fourgram = three_prev[1] + " " + three_prev[1] + " " + token + three_next[0]
    right_center_fourgram = three_prev[0]+ " " + token + " " + three_next[0] + " " + three_next[1]
    right_fourgram =   token + " " + three_next[0] + " " + three_next[1] + " " + three_next[2]

    n_grams = [bigram, reverse_bigram, trigram, left_trigram, right_trigram, left_fourgram, left_center_fourgram, right_center_fourgram, right_fourgram]

    if any(expr in multi_negations for expr in n_grams): # check if any of the n-grams matches a multiword negation expression
        return 1
    else:
        return 0

def is_punctuation(token):
    '''Return 1 if the token is a punctuation character, 0 otherwise'''
    if token in string.punctuation:
        return 1
    return 0

def generate_lexicon(doc,  n=5):
    '''Generates the lexicon used in the substring-count feature. This lexicon is a counter object
        which contains the token itself and the word initial n-gram up to 5 characters.'''
    lexicon = Counter()
    token_pos_list = [(token.text, token.tag_) for token in doc]
    lexicon.update(token_pos_list)
    for token, tag in token_pos_list:
        if len(token) < n:
            n_new = len(token)
        else:
            n_new = n
        ngrams = generate_ngrams(token, n_new)[0] # start ngrams
        gram_pos_list = [(gram, tag) for gram in ngrams]
        lexicon.update(gram_pos_list)
    return lexicon

def generate_ngrams(token, n=5):
    '''Function that generates word-initial and word-end character n-grams for a token.
        :returns: start_igrams: a list of word initial character ngrams
        :returns: end_igrams: a list of word final character ngrams'''
    start_igrams = []
    end_igrams = []

    for i in reversed(range(1, n+1)):

        igrams = [''.join(gram) for gram in ngrams(token, i)]
        start_igrams.append(igrams[0])
        end_igrams.append(igrams[-1])

    return start_igrams, end_igrams

def create_ngram_features(token, POS, lexicon, n=5):
    '''Function that return the ngrams and the substring count feature
        :param: token: the token (str)
        :param: lexicon: the lexicon of tokens in the corpus
        :returns: start_igrams: the word initial character ngrams
        :returns: end_igrams: the word final character ngrams
        :returns: substring_count: the number of occurences of the substring in the corpus'''
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
        substring_count = 0
        for gram in start_ngrams:
            substring_count += lexicon[(gram, POS)]
        return start_ngrams, end_ngrams, substring_count# first (longest start ngram of affixless token is the match we search for)
    if len(token) < n:
        n = len(token)
    start_ngrams, end_ngrams = generate_ngrams(token, n)
    return start_ngrams, end_ngrams, 0

import csv
from nltk import ngrams
from collections import Counter
import string
from utils import *

neg_prefixed = ['un', 'in', 'im', 'il', 'dis', 'ir', 'non']
neg_suffixed = ['less']

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
        return start_ngrams, end_ngrams, lexicon[affixless_token]
    if len(token) < n:
        n = len(token)
    start_ngrams, end_ngrams = generate_ngrams(token, n)
    return start_ngrams, end_ngrams, 0

def main(args=None):
    if not args:
        args = sys.argv
    path = args[1]
    conll = read_in_conll_file(path)
    tokenlist = list_of_tokens(conll)
    lexicon = generate_lexicon(tokenlist)
    print(lexicon)
    for token in tokenlist[:40000]:
        n = 5
        features = create_ngram_features(token, lexicon, n)
        n_gram_feature = features[0]
        lexicon_feature = features[1]
        #print(n_gram_feature, lexicon_feature)


args = ['x', r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\bioscope-corpus\bioscope.papers.columns.txt']
main(args)

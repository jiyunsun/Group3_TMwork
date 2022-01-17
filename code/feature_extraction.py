from read_corpus import *
from nltk import ngrams
from collections import Counter

neg_prefixed = ['un', 'in', 'im', 'il', 'dis']

def generate_ngrams(token, n):
    n_grams = [[],[]]
    for i in reversed(range(1, n+1)):
        igrams = [''.join(gram) for gram in ngrams(token, i)]
        start_igram = igrams[0]
        end_igram = igrams[-1]
        n_grams[0].append(start_igram)
        n_grams[1].append(end_igram)
    return n_grams

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

def create_ngram_features(token, lexicon, n=5):
    for prefix in neg_prefixed:
        if token.startswith(prefix):
            prefixless_token = token.replace(prefix,'')
            if len(prefixless_token) < n:
                n = len(prefixless_token)

            if prefixless_token:
                #print(prefix, prefixless_token, end=' ')
                grams = generate_ngrams(prefixless_token, n)
                #print(grams)
                return grams, lexicon[prefixless_token]
    return ['_', '_']


def main(args=None):
    if not args:
        args = sys.argv
    path = args[1]
    conll = read_in_conll_file(path)
    tokenlist = list_of_tokens(conll)
    lexicon = generate_lexicon(tokenlist)
    print(lexicon)
    for token in tokenlist:
        n = 5
        features = create_ngram_features(token, lexicon, n)
        n_gram_feature = features[0]
        lexicon_feature = features[1]
        print(n_gram_feature, lexicon_feature)


args = ['x', r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\bioscope-corpus\bioscope.papers.columns.txt']
main(args)

from utils import *
import numpy
import gensim

num_features = 300
print('Loading embeddings')
embeddings_path = r'C:\Users\Tessel Wisman\Documents\TextMining\GoogleNews-vectors-negative300\GoogleNews-vectors-negative300.bin'
word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=True)
print('Done')


def extract_embeddings_as_features_and_gold(conllfile, word_embedding_model, embedding_size):
    '''
    THIS CODE WAS ADAPTED FROM THE ML4NLP COURSE
    Function that extracts features and gold labels using word embeddings

    :param conllfile: path to conll file
    :param word_embedding_model: a pretrained word embedding model
    :type conllfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors

    :return features: list of vector representation of tokens
    :return labels: list of gold labels
    '''
    labels = []
    features = []

    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter ='\t',quotechar ='|')
    for row in csvreader:
        if len(row) > 0: #check for cases where empty lines mark sentence boundaries (which some conll files do).
            lemma = row[4]
            if lemma in word_embedding_model:
                vector = word_embedding_model[lemma]
            else:
                vector = [0] * 300 #TESSEL I'M NOT SURE ABOUT THAT, SHOULD BE [3] TOO?
            features.append(vector)
            labels.append(row[-1])
    return features, labels

def extract_embeddings_as_features(conllfile, word_embedding_model, embedding_size):
    '''
    THIS CODE WAS ADAPTED FROM THE ML4NLP COURSE
    Function that extracts features and gold labels using word embeddings

    :param conllfile: path to conll file
    :param word_embedding_model: a pretrained word embedding model
    :type conllfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors

    :return features: list of vector representation of tokens
    :return labels: list of gold labels
    '''
    labels = []
    features = []

    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter ='\t',quotechar ='|')
    for row in csvreader:
        if len(row) > 0: #check for cases where empty lines mark sentence boundaries (which some conll files do).
            lemma = row[4]
            if lemma in word_embedding_model:
                vector = word_embedding_model[lemma]
            else:
                vector = [0] * 300 #TESSEL I'M NOT SURE ABOUT THAT, SHOULD BE [3] TOO?
            features.append(vector)
    return features

def main(args=None):
    if not args:
        args = sys.argv
    path = args[1]
    extract_embeddings_as_features_and_gold(path, word_embedding_model, num_features)

args = ['x', r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\SEM-2012-SharedTask-CD-SCO-simple.v2\SEM-2012-SharedTask-CD-SCO-training-simple.v2-preprocessed.txt']
main(args)

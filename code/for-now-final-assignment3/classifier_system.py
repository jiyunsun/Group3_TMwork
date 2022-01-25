from utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np
import sys
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
import csv
import random
import gensim
import os


feature_list = ['token', 'lemma', 'prev', 'next','pos', 'punct', 'prefix', 'suffix', 'infix', 'sbs_count', 'match_one', 'match_multi', 'prev_tok_cue']
feature_to_index = {'token': 3, 'lemma':4, 'prev':5, 'next':6,'pos':7, 'punct':8, 'prefix':9, 'suffix':10, 'infix':11, 'sbs_count':12, 'match_one':13, 'match_multi':14, 'prev_tok_cue':15}


def classify_data(model, vec, selected_features,inputdata, outputfile):
    '''
    THIS CODE WAS ADAPTED FROM THE ML4NLP COURSE
    Function that performs the named entity recognition and writes an output file that is the same as the input file
        except classification result is added at the end of each sample row.
        params:
            model: a fitted LogisticRegression model
            vec: a fitted DictVectorizer
            inputdata: input data file to be classified
            outputfile: file to write output

    '''
    features = extract_features(inputdata,selected_features)
    print('features extracted')
    features = vec.transform(features)
    predictions = model.predict(features)
    print('model predicted')
    outfile = open(outputfile, 'w')
    header = feature_list + ['negcuelabel\n']
    outfile.write('\t'.join(header))
    counter = 0
    with open(inputdata, 'r') as infile:
        next(infile)
        for line in infile:
            stripped_line = line.rstrip('\n')

            if len(stripped_line.split()) > 0:

                outfile.write(stripped_line + '\t' + predictions[counter] + '\n')
                counter += 1
    outfile.close()

def create_classifier(train_features, train_targets, classifier_type):
    '''
    THIS CODE WAS ADAPTED FROM THE ML4NLP COURSE
     Creates a Logistic Regression classifier and fits features and targets
        params:
            train_features: list of dicts {'token' : TOKEN}
            train_targets: list of targets
        returns:
            model: the fitted logistic regression model
            vec: the dict vectorizer object used to transform the train_features
    '''
    if classifier_type == 'NB':
        classifier= ComplementNB()
    elif classifier_type == 'SVM':
        classifier = LinearSVC()
    elif classifier_type == 'logreg':
        classifier = LogisticRegression(max_iter=2000)

    vec = DictVectorizer()
    print('fitting', classifier_type)
    features_vectorized = vec.fit_transform(train_features)

    #print(type(features_vectorized))
    #print(np.asarray(features_vectorized).shape)
    model = classifier.fit(features_vectorized, train_targets)

    return model, vec

def classify_embeddings_data(model, inputdata, outputfile, word_embedding_model, vectorizer, num_features, selected_features):
    ''' Function that performs the named entity recognition and writes an output file that is the same as the input file
        except classification result is added at the end of each sample row.
        params:
            model: a fitted LogisticRegression model
            vec: a fitted DictVectorizer
            inputdata: input data file to be classified
            outputfile: file to write output
            word_embedding_model: the we model
            num_features: number of features used in the model

    '''
    features = extract_embeddings_as_features(inputdata, word_embedding_model, num_features, selected_features, vectorizer)
    predictions = model.predict(features)
    outfile = open(outputfile, 'w')
    header = feature_list + ['negcuelabel\n']
    outfile.write('\t'.join(header))
    counter=0
    with open(inputdata, 'r') as infile:
        next(infile)
        for line in infile:
            stripped_line = line.rstrip('\n')

            if len(stripped_line.split()) > 0:
                outfile.write(stripped_line + '\t' + predictions[counter] + '\n')
                counter += 1
    outfile.close()

def create_embeddings_classifier(train_features, train_targets, classifier_type):
    ''' Creates a classifier based on word embeddings and fits features and targets
        params:
            train_features: list of dicts {'token' : TOKEN}
            train_targets: list of targets
            classifier_type: what classifier to use (str)
        returns:
            model: the fitted logistic regression model
    '''
    if classifier_type == 'NB':
        # https://stackoverflow.com/questions/67224279/how-can-i-resolve-this-error-valueerror-negative-values-in-data-passed-to-m

        classifier= ComplementNB()
    elif classifier_type == 'SVM':
        classifier = LinearSVC()
    elif classifier_type == 'logreg':
        classifier = LogisticRegression(max_iter=3000)

    model = classifier.fit(train_features, train_targets)

    return model

def extract_feature_values(row, selected_features):
    '''
    THIS CODE WAS ADAPTED FROM THE ML4NLP COURSE
    Function that extracts feature value pairs from row

    :param row: row from conll file
    :param selected_features: list of selected features
    :type row: string
    :type selected_features: list of strings

    :returns: dictionary of feature value pairs
    '''
    feature_values = {}
    for feature_name in selected_features:
        r_index = feature_to_index.get(feature_name)
        feature_values[feature_name] = row[r_index]

    return feature_values

def combine_sparse_and_dense_features(dense_vectors, sparse_features):
    '''
    THIS CODE WAS ADAPTED FROM THE ML4NLP COURSE
    Function that takes sparse and dense feature representations and appends their vector representation

    :param dense_vectors: list of dense vector representations
    :param sparse_features: list of sparse vector representations
    :type dense_vector: list of arrays
    :type sparse_features: list of lists

    :returns: list of arrays in which sparse and dense vectors are concatenated
    '''

    combined_vectors = []
    sparse_vectors = np.array(sparse_features.toarray())
    #print(sparse_vectors)
    if sparse_vectors.size <=0:
        return dense_vectors
    for index, vector in enumerate(sparse_vectors):
        combined_vector = np.concatenate((vector,dense_vectors[index]))
        combined_vectors.append(combined_vector)
    #print(combined_vectors[0].shape)
    return combined_vectors

def extract_features_and_labels(trainingfile, selected_features):
    '''
    THIS CODE WAS ADAPTED FROM THE ML4NLP COURSE
    Extracts the features and labels from the gold datafile.
        params:
            trainingfile: the .conll file with samples on each line. First element in the line is the token,
                final element is the label
        returns:
            data: list of dicts {'token': TOKEN}
            targets: list of target names for the tokens
            '''
    data = []
    targets = []
    with open(trainingfile, 'r', encoding='utf8') as infile:
        next(infile)
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                feature_dict = extract_feature_values(components, selected_features)

                data.append(feature_dict)
                #gold is in the last column
                targets.append(components[-1])
    return data, targets

def extract_features(inputfile, selected_features):
    '''
    THIS CODE WAS ADAPTED FROM THE ML4NLP COURSE
    Similar to extract_'features_and_labels, but only extracts data
        params:
            inputfile: an input file containing samples on each row, where feature token is the first word on each row
        returns:
            data: a list of dicts ('token': TOKEN)
            '''
    data = []
    with open(inputfile, 'r', encoding='utf8') as infile:
        next(infile)
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                feature_dict = extract_feature_values(components, selected_features)
                data.append(feature_dict)
    return data

def extract_word_embedding(token, word_embedding_model, num_features):
    '''
    THIS CODE WAS ADAPTED FROM THE ML4NLP COURSE
    Function that returns the word embedding for a given token out of a distributional semantic model and a 300-dimension vector of 0s otherwise

    :param token: the token
    :param word_embedding_model: the distributional semantic model
    :type token: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors

    :returns a vector representation of the token
    '''
    if token in word_embedding_model:
        vector = word_embedding_model[token]
    else:
        vector = [0]*num_features
    return vector

def extract_embeddings_as_features_and_gold(conllfile, word_embedding_model, num_features, selected_features, vectorizer=None):
    '''
    THIS CODE WAS ADAPTED FROM THE ML4NLP COURSE
    Function that extracts features and gold labels using word embeddings

    :param conllfile: path to conll file
    :param word_embedding_model: a pretrained word embedding model
    :param num_features: number of features in model
    :type conllfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    :type num_features: int

    :return features: list of vector representation of tokens
    :return labels: list of gold labels
    '''
    ### This code was partially inspired by code included in the HLT course, obtained from https://github.com/cltl/ma-hlt-labs/, accessed in May 2020.
    labels = []
    dense_features = []
    traditional_features = []
    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    other_feats_to_extract = [s for s in selected_features if s != 'prev' and s!= 'next']
    #print(other_feats_to_extract)
    next(csvreader)
    for row in csvreader:
        #check for cases where empty lines mark sentence boundaries (which some conll files do).

        if len(row) > 0:
            lemma_id = feature_to_index['lemma']
            prev_id = feature_to_index['prev']
            next_id = feature_to_index['next']
            token_vector = extract_word_embedding(row[lemma_id], word_embedding_model, num_features)
            if 'prev' in selected_features and 'next' in selected_features:
                pt_vector = extract_word_embedding(row[prev_id], word_embedding_model, num_features)
                nt_vector = extract_word_embedding(row[next_id], word_embedding_model, num_features)
                dense_features.append(np.concatenate((token_vector,pt_vector, nt_vector)))
            elif 'prev' in selected_features:
                pt_vector = extract_word_embedding(row[prev_id], word_embedding_model, num_features)
                dense_features.append(np.concatenate((token_vector,pt_vector)))
            elif 'next' in selected_features:
                nt_vector = extract_word_embedding(row[next_id], word_embedding_model, num_features)
                dense_features.append(np.concatenate((token_vector,nt_vector)))
            else:
                dense_features.append(token_vector)
            if other_feats_to_extract != None:
                other_features = extract_feature_values(row, other_feats_to_extract)
                traditional_features.append(other_features)
            labels.append(row[-1])
    if vectorizer == None:
        vectorizer = DictVectorizer()
        vectorizer.fit(traditional_features)

    sparse_features = vectorizer.transform(traditional_features)
    combined_vectors = combine_sparse_and_dense_features(dense_features, sparse_features)
    return combined_vectors, labels, vectorizer

def extract_embeddings_as_features(conllfile,word_embedding_model, num_features, selected_features, vectorizer=None):
    '''
    THIS CODE WAS ADAPTED FROM THE ML4NLP COURSE
    Function that extracts features and gold labels using word embeddings

    :param conllfile: path to conll file
    :param word_embedding_model: a pretrained word embedding model
    :param num_features: number of features in model
    :type conllfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    :type num_features: int

    :return features: list of vector representation of tokens
    :return labels: list of gold labels
    '''
    ### This code was partially inspired by code included in the HLT course, obtained from https://github.com/cltl/ma-hlt-labs/, accessed in May 2020.
    dense_features = []
    traditional_features = []
    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    other_feats_to_extract = [s for s in selected_features if s != 'prev' and s!= 'next']
    print(other_feats_to_extract)
    next(csvreader)
    for row in csvreader:
        #check for cases where empty lines mark sentence boundaries (which some conll files do).
        if len(row) > 0:
            lemma_id = feature_to_index['lemma']
            prev_id = feature_to_index['prev']
            next_id = feature_to_index['next']
            token_vector = extract_word_embedding(row[lemma_id], word_embedding_model, num_features)
            if 'prev' in selected_features and 'next' in selected_features:
                pt_vector = extract_word_embedding(row[prev_id], word_embedding_model, num_features)
                nt_vector = extract_word_embedding(row[next_id], word_embedding_model, num_features)
                dense_features.append(np.concatenate((token_vector,pt_vector, nt_vector)))
            elif 'prev' in selected_features:
                pt_vector = extract_word_embedding(row[prev_id], word_embedding_model, num_features)
                dense_features.append(np.concatenate((token_vector,pt_vector)))
            elif 'next' in selected_features:
                nt_vector = extract_word_embedding(row[next_id], word_embedding_model, num_features)
                dense_features.append(np.concatenate((token_vector,nt_vector)))
            else:
                dense_features.append(token_vector)
            if other_feats_to_extract != None:
                other_features = extract_feature_values(row, other_feats_to_extract)
                traditional_features.append(other_features)
    if vectorizer == None:
        vectorizer = DictVectorizer()
        vectorizer.fit(traditional_features)
    sparse_features = vectorizer.transform(traditional_features)
    combined_vectors = combine_sparse_and_dense_features(dense_features, sparse_features)
    return combined_vectors

def main(argv=None):

    if argv is None:
        argv = sys.argv

    trainingfile = r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\SEM-2012-SharedTask-CD-SCO-simple.v2\SEM-2012-SharedTask-CD-SCO-training-simple.v2-preprocessed.txt'
    inputfile = r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\SEM-2012-SharedTask-CD-SCO-simple.v2\SEM-2012-SharedTask-CD-SCO-dev-simple.v2-preprocessed.txt'
    output_basepath = r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\Group3_TMWork\models\250122'

    feature_ablation = argv[1]
    embeddings = argv[2]

    print('Starting training')
    print('Doing feature ablation:', feature_ablation )
     ### this model has 300 dimensions so we set the number of features to 300

    # These are all feature combinations that were tested in the feature ablation
    feature_selections = [['token']] # EDIT TO LIST OF LISTS OF OUR DESIRED FEATURES
    # If we don't want to run the ablation, the standard system is run with the basics
    if not feature_ablation:
        feature_selections = [['token']]

    if embeddings:
        num_features = 300
        print('Loading embeddings')
        embeddings_path = r'C:\Users\Tessel Wisman\Documents\TextMining\GoogleNews-vectors-negative300\GoogleNews-vectors-negative300.bin'
        word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=True)
        print('Done')

    for model in ['SVM', 'logreg']:
        print('Model used:', model)
        for feature_select in feature_selections:
            print('Features selected in this run:',feature_select)

            training_features, gold_labels = extract_features_and_labels(trainingfile, selected_features=feature_select)
            ml_model, vec = create_classifier(training_features, gold_labels, model)
            classify_data(ml_model, vec, feature_select, inputfile, output_basepath + '_' + model + '_'.join(feature_select) + '.txt')
            print('finished training the ', model, 'model on', feature_select )
            if embeddings:
                feature_select_emb = [f for f in feature_select if f != 'token' and f!= 'prev' and f!= 'next']
                embedding_features, embedding_gold_labels, vec = extract_embeddings_as_features_and_gold(trainingfile, word_embedding_model, num_features, feature_select_emb)
                print('starting training with word embeddings')
                ml_model_emb = create_embeddings_classifier(embedding_features, embedding_gold_labels, model)
                classify_embeddings_data(ml_model_emb, inputfile, output_basepath + '_WE_' + model + '_'.join(feature_select) + '.txt', word_embedding_model, vec, num_features, feature_select_emb)
                print('finished with word embeddings for', model, 'on', feature_select)

#args = ['python','../../data/conll2003_ret.train-preprocessed_with_feats.conll', '../../data/conll2003_ret.test-preprocessed_chunks.conll', '../../models/1612_cl_fa_non_scaled_', r'C:\Users\Tessel Wisman\Documents\TextMining\GoogleNews-vectors-negative300\GoogleNews-vectors-negative300.bin', False, True]
#main(args)
#if __name__ == '__main__':
#    main()

args = ['x', True, True]
main(args)

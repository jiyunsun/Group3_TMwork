from utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import sys
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
import csv
import random
import gensim
import os
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer

#### Most functions in this script were adapted from the ML4NLP course submission ####
embeddings_path = r'C:\Users\Tessel Wisman\Documents\TextMining\GoogleNews-vectors-negative300\GoogleNews-vectors-negative300.bin' # CHANGE TO LOCAL PATH
feature_list = ['token', 'lemma', 'prev', 'next','pos', 'punct', 'prefix', 'suffix', 'infix', 'sbs_count',
        'match_one', 'match_multi', 'prev_tok_cue'] # GLOBAL list of features
feature_to_index = {'token': 3, 'lemma':4, 'prev':5, 'next':6,'pos':7, 'punct':8,
    'prefix':9, 'suffix':10, 'infix':11, 'prev_tok_cue':12,'sbs_count':13, 'match_one':14, 'match_multi':15} # MAPPING of features to column idx in files


def classify_data(model, vec, selected_features,inputdata, outputfile, model_type):
    '''
    Function that performs the named entity recognition and writes an output file that is the same as the input file
        except classification result is added at the end of each sample row.
        param: model: a fitted LogisticRegression model
        param: vec: a fitted DictVectorizer
        param: inputdata: input data file to be classified
        param: outputfile: file to write output
        param: model_type: which classifier is used (CRF or SVM)

    '''
    if model_type == 'CRF':
        features = extract_CRF_features(inputdata,selected_features) # we need a different way to extract features when using CRF
    else:
        features = extract_features(inputdata,selected_features)
        features = vec.transform(features)

    predictions = model.predict(features) # predict classifications

    if model_type == 'CRF': # if CRF is used, we need to align results with our input and put it in lists of sentences
        predictions = [prediction for sent in predictions for prediction in sent]

    outfile = open(outputfile, 'w')
    header = feature_list + ['negcuelabel\n']
    outfile.write('\t'.join(header))

    counter = 0

    with open(inputdata, 'r') as infile:
        next(infile) # skip header
        for line in infile:
            stripped_line = line.rstrip('\n')
            if len(stripped_line.split()) > 0: # check if not empty
                outfile.write(stripped_line + '\t' + predictions[counter] + '\n') # add prediction to file
                counter += 1
    outfile.close()

def create_classifier(train_features, train_targets, classifier_type, grid_search=False):
    '''
     Creates a Logistic Regression classifier and fits features and targets
        param: train_features: list of dicts {'token' : TOKEN}
        param: train_targets: list of targets
        param:classifier_type: classifier used (CRF or SVM)
        param: grid_search: to do grid search or not (Boolean value)
        returns: model: the fitted logistic regression model
        returns: vec: the dict vectorizer object used to transform the train_features

    '''
    if classifier_type == 'SVM':
        classifier = LinearSVC(max_iter=4000)
        parameters = dict(C = (0.01, 0.1, 1.0), loss= ('hinge', 'squared_hinge'), tol=(1e-4, 1e-3, 1e-2, 1e-1)) # parameters for grid search

        vec = DictVectorizer()
        features_vectorized = vec.fit_transform(train_features)
    elif classifier_type == 'CRF':
        classifier = CRF(algorithm='l2sgd', all_possible_transitions=True) # set standard mparams
        parameters = dict(algorithm= ('lbfgs', 'l2sgd'), all_possible_transitions=(True, False))
        features_vectorized = train_features
        vec = None

    if grid_search:
        f1_scorer = make_scorer(metrics.flat_f1_score, average='macro')
        grid = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='f1_scorer')
        model = grid.fit(features_vectorized, train_targets)
        print('Parameters', grid.best_params_)
        classifier = grid.best_estimator_
        return model, vec
    else:
        # if we don't do grid search, just fit the model
        model = classifier.fit(features_vectorized, train_targets)
        return model, vec

def classify_embeddings_data(model, inputdata, outputfile, word_embedding_model, vectorizer, num_features, selected_features, model_type):
    ''' Function that performs the named entity recognition and writes an output file that is the same as the input file
        except classification result is added at the end of each sample row.
        param: model: a fitted LogisticRegression model
        param: inputdata: input data file to be classified
        param: outputfile: file to write output
        param: word_embedding_model: the WE model
        param: vectorizer: DictVectorizer object
        param: num_features: number of features used in the model
        param: selected_features: selection of features used in current model (list)
        param: model_type: classifier name (str) which will be CRF or SVM

    '''
    features = extract_embeddings_as_features(inputdata,word_embedding_model, num_features, selected_features, vectorizer=vectorizer)
    predictions = model.predict(features)
    if model_type == 'CRF': # to sentence format
        predictions = [prediction for sent in predictions for prediction in sent]

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
        param: train_features: list of dicts {'token' : TOKEN}
        param: train_targets: list of targets
        param: classifier_type: what classifier to use (str)
        returns: model: the fitted logistic regression model
    '''
    if classifier_type == 'SVM':
        classifier = LinearSVC(max_iter=4000)
    else:
        print('Invalid classifier type. Can only use SVM')
        return False

    model = classifier.fit(train_features, train_targets)

    return model

def extract_feature_values(row, selected_features):
    '''
    Function that extracts feature value pairs from row and puts them in a dict

    :param row: row from conll file (list)
    :param selected_features: list of selected features

    :returns: dictionary of feature value pairs
    '''
    feature_values = {}
    for feature_name in selected_features:
        r_index = feature_to_index.get(feature_name)
        feature_values[feature_name] = row[r_index]

    return feature_values

def combine_sparse_and_dense_features(dense_vectors, sparse_features):
    '''
    Function that takes sparse and dense feature representations and appends their vector representation

    :param dense_vectors: list of dense vector representations (list of arrays)
    :param sparse_features: list of sparse vector representations (list of lists)
    :returns: list of arrays in which sparse and dense vectors are concatenated
    '''

    combined_vectors = []
    sparse_vectors = np.array(sparse_features.toarray())

    if sparse_vectors.size <=0: # if there are no sparse vectors, abort and return dense
        return dense_vectors
    for index, vector in enumerate(sparse_vectors):
        combined_vector = np.concatenate((vector,dense_vectors[index]))
        combined_vectors.append(combined_vector)
    return combined_vectors

def extract_CRF_features(inputfile, selected_features):
    '''
    Similar to extract_features_and_labels, but only extracts data (catered to CRF architecture
    by putting the data in lists of lists(sentences) of dicts)
        param: inputfile: an input file containing samples on each row, where feature token is the first word on each row
        param: selected_features: the list of selected features in this model
        returns: data: a list of lists dicts ('token': TOKEN)
            '''
    data = []
    current_sent = []
    with open(inputfile, 'r', encoding='utf8') as infile:
        next(infile)
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                feature_dict = extract_feature_values(components, selected_features)
                current_sent.append(feature_dict)
                if components[3] == '.':
                    data.append(current_sent)
                    current_sent = []

    return data

def extract_CRF_features_and_labels(trainingfile, selected_features):
    '''
    (catered to CRF architecture
    by putting the data in lists of lists(sentences) of dicts)
    Extracts the features and labels from the gold datafile.
        param: trainingfile: the .conll file with samples on each line. First element in the line is the token,
                final element is the label
        param: selected_features: the list of selected features
        returns: data: list of list dicts {'token': TOKEN}
        returns: targets: list of target names for the tokens
            '''
    current_sent = []
    current_targets = []
    data = []
    targets = []
    with open(trainingfile, 'r', encoding='utf8') as infile:
        next(infile)
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                feature_dict = extract_feature_values(components, selected_features)

                current_sent.append(feature_dict)
                current_targets.append(components[-1])
                if components[3] == '.':
                    data.append(current_sent)
                    current_sent = []
                #gold is in the last column
                    targets.append(current_targets)
                    current_targets = []
    return data, targets

def extract_features_and_labels(trainingfile, selected_features):
    '''
    Extracts the features and labels from the gold datafile.
        param: trainingfile: the .conll file with samples on each line. First element in the line is the token,
                final element is the label
        param: selected_features: list of selected features
        returns: data: list of dicts {'token': TOKEN}
        returns: targets: list of target names for the tokens
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
    Similar to extract_'features_and_labels, but only extracts data
        params: inputfile: an input file containing samples on each row, where feature token is the first word on each row
        param: selected_features: list of selected features
        returns: data: a list of dicts ('token': TOKEN)
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
    Function that returns the word embedding for a given token out of a distributional semantic model and a 300-dimension vector of 0s otherwise

    :param token: the token
    :param word_embedding_model: the distributional semantic model
    :param: num_features: number of features in the embedding model (n)

    :returns a vector representation of the token
    '''
    if token in word_embedding_model:
        vector = word_embedding_model[token]
    else:
        vector = [0]*num_features
    return vector

def extract_embeddings_as_features_and_gold(conllfile, word_embedding_model, num_features, selected_features, vectorizer=None):
    '''
    Function that extracts features and gold labels using word embeddings

    :param conllfile: path to conll file
    :param word_embedding_model: a pretrained word embedding model
    :param num_features: number of features in model
    :param selected_features: list of selected features
    :param vectorizer: optional to pass an already fitted DictVectorizer

    :return combined_vectors: list of vector representations of tokens
    :return labels: list of gold labels
    :return vectorizer: the fitted vectorizer object
    '''
    ### This code was partially inspired by code included in the HLT course, obtained from https://github.com/cltl/ma-hlt-labs/, accessed in May 2020.
    labels = []
    dense_features = []
    traditional_features = []
    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    other_feats_to_extract = [s for s in selected_features if s != 'prev' and s!= 'next' and s != 'token' and s!='lemma'] # we will extract these features as embeddings, so we want to exclude them from our list of one-hot features

    next(csvreader)
    for row in csvreader:

        if len(row) > 0:
            lemma_id = feature_to_index['lemma']
            prev_id = feature_to_index['prev']
            next_id = feature_to_index['next']
            token_vector = extract_word_embedding(row[lemma_id], word_embedding_model, num_features) # extract lemma vector
            if 'prev' in selected_features and 'next' in selected_features: # if we have previous and next, combine them all
                pt_vector = extract_word_embedding(row[prev_id], word_embedding_model, num_features)
                nt_vector = extract_word_embedding(row[next_id], word_embedding_model, num_features)
                dense_features.append(np.concatenate((token_vector,pt_vector, nt_vector)))
            elif 'prev' in selected_features: # if we want to include previous token, combine it with the lemma vector
                pt_vector = extract_word_embedding(row[prev_id], word_embedding_model, num_features)
                dense_features.append(np.concatenate((token_vector,pt_vector)))
            elif 'next' in selected_features: # same for next token
                nt_vector = extract_word_embedding(row[next_id], word_embedding_model, num_features)
                dense_features.append(np.concatenate((token_vector,nt_vector)))
            else: # if we only want the lemma
                dense_features.append(token_vector)
            if other_feats_to_extract != None: # if there are any other features to extract, get them
                other_features = extract_feature_values(row, other_feats_to_extract)
                traditional_features.append(other_features)
            labels.append(row[-1])
    if vectorizer == None: # vectorize the traditional features
        vectorizer = DictVectorizer()
        vectorizer.fit(traditional_features)

    sparse_features = vectorizer.transform(traditional_features)

    combined_vectors = combine_sparse_and_dense_features(dense_features, sparse_features) #combine sparse and dense

    return combined_vectors, labels, vectorizer

def extract_embeddings_as_features(conllfile,word_embedding_model, num_features, selected_features, vectorizer=None):
    '''
    Function that extracts features using word embeddings. Works similar to extract_embeddings_as_features_and_gold, only skips extracting labels

    :param conllfile: path to conll file
    :param word_embedding_model: a pretrained word embedding model
    :param num_features: number of features in model
    :param selected_features: list of selected features
    :param vectorizer: optional parameter to pass a fitted DictVectorizer

    :return combined_vectors: the feature vectors
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

    trainingfile = argv[1]#r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\SEM-2012-SharedTask-CD-SCO-simple.v2\SEM-2012-SharedTask-CD-SCO-training-simple.v2-preprocessed.txt'
    inputfile = argv[2]#r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\SEM-2012-SharedTask-CD-SCO-simple.v2\SEM-2012-SharedTask-CD-SCO-test-preprocessed.txt'
    output_basepath = argv[3]#r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\Group3_TMWork\models\testrun010222'
    feature_ablation = argv[4]
    embeddings = argv[5]

    print('Starting training')
    print('Doing feature ablation?', feature_ablation )


    # These are all feature combinations that were tested in the feature ablation
    feature_selections =[['lemma'],
    ['lemma', 'prev_tok_cue'], ['lemma', 'prev'], # is expected to up the performance for SVM, but not for CRF
    ['lemma', 'prev', 'next'],
    ['lemma', 'prev', 'next', 'prev_tok_cue'],
    ['token','lemma', 'prev', 'next'],
    ['token', 'lemma', 'prev', 'next', 'prev_tok_cue'], # is there a value in adding both previous and next token?
    ['lemma', 'match_one', 'match_multi'], # how does this compare to rule-matching single- and multiword tokens?

    [ 'lemma', 'prefix', 'suffix', 'infix'],
    [ 'lemma', 'infix', 'sbs_count'], # affix features versus Lapponi feature
    [ 'lemma', 'prefix', 'suffix', 'infix', 'sbs_count'], # combination of both
    ['lemma', 'pos', 'prefix', 'suffix', 'infix'],
    ['lemma', 'pos', 'prefix', 'suffix', 'infix', 'sbs_count'],
    [ 'lemma', 'prefix', 'suffix', 'infix', 'prev_tok_cue', 'match_one'],#,
    ['lemma', 'pos', 'prefix', 'suffix', 'infix', 'sbs_count', 'match_one', 'match_multi'], # add rule-based + affixal to check basic performance
    feature_list, # all features at once
    #also add one by one
    ['token', 'lemma'],['token', 'lemma', 'prev'], ['token', 'lemma', 'prev','next'], ['token', 'lemma', 'prev','next','pos'],
    ['token', 'lemma', 'prev','next','pos', 'punct'], ['token', 'lemma', 'prev','next','pos', 'punct', 'prefix'],
    ['token', 'lemma', 'prev','next','pos', 'punct', 'prefix','suffix'], ['token', 'lemma', 'prev','next','pos', 'punct', 'prefix','suffix','infix'],
    ['token', 'lemma', 'prev','next','pos', 'punct', 'prefix','suffix','infix', 'sbs_count'],
    ['token', 'lemma', 'prev','next','pos', 'punct', 'prefix','suffix','infix', 'sbs_count', 'match_one'],
    ['token', 'lemma', 'prev','next','pos', 'punct', 'prefix','suffix','infix', 'sbs_count', 'match_one', 'match_multi'],
    ['token', 'lemma', 'prev','next','pos', 'punct', 'prefix','suffix','infix', 'sbs_count', 'match_one', 'match_multi','prev_tok_cue']]
    # If we don't want to run the ablation, the standard system is run with all the features
    if not feature_ablation:
        feature_selections = [feature_list]

    if embeddings: # we only load the embeddings if they are used in this run
        num_features = 300      ### this model has 300 dimensions so we set the number of features to 300
        print('Loading embeddings')
        word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=True)
        print('Done')

    for model in ['SVM', 'CRF']:
        print('Model used:', model)
        for feature_select in feature_selections:
            print('Features selected in this run:',feature_select)
            #Load the features
            if model == 'CRF':
                training_features, gold_labels = extract_CRF_features_and_labels(trainingfile, selected_features=feature_select)
            else:
                training_features, gold_labels = extract_features_and_labels(trainingfile, selected_features=feature_select)
            # classify
            ml_model, vec = create_classifier(training_features, gold_labels, model)
            classify_data(ml_model, vec, feature_select, inputfile, output_basepath + '_' + model + '_'.join(feature_select) + '.txt', model)
            print('finished training the ', model, 'model on', feature_select )
            # Only executed if specified that we want to run embeddings and we are not doing CRF
            if embeddings and model != 'CRF':
                embedding_features, embedding_gold_labels, vec = extract_embeddings_as_features_and_gold(trainingfile, word_embedding_model, num_features, feature_select)
                print('starting training with word embeddings')
                ml_model_emb = create_embeddings_classifier(embedding_features, embedding_gold_labels, model)
                classify_embeddings_data(ml_model_emb, inputfile, output_basepath + '_WE_' + model + '_'.join(feature_select) + '.txt', word_embedding_model, vec, num_features, feature_select, model)
                print('finished with word embeddings for', model, 'on', feature_select)

#args = ['python','../../data/conll2003_ret.train-preprocessed_with_feats.conll', '../../data/conll2003_ret.test-preprocessed_chunks.conll', '../../models/1612_cl_fa_non_scaled_', r'C:\Users\Tessel Wisman\Documents\TextMining\GoogleNews-vectors-negative300\GoogleNews-vectors-negative300.bin', False, True]
#main(args)
if __name__ == '__main__':
    main()
#import os
#os.chdir(r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\Group3_TMwork\code\\')
#args = ['x', '..\SEM-2012-corpus\SEM-2012-training-preprocessed.txt', '..\SEM-2012-corpus\SEM-2012-dev-preprocessed.txt', '..\models\\1222',  True, True]
#main(args)

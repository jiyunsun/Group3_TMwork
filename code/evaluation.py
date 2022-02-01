import sys
import pandas as pd
import os
from collections import defaultdict, Counter
import glob
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
## THIS SCRIPT WAS ADAPTED FROM THE ML4NLP COURSE

def extract_annotations(inputfile, delimiter='\t'):
    '''
    This function extracts annotations represented in the conll format from a file

    :param inputfile: the path to the conll file (str)
    :param delimiter: optional parameter to overwrite the default delimiter (tab) (str)
    :returns: the annotations as a list
    '''
    conll_input = pd.read_csv(inputfile, sep=delimiter)
    annotations = conll_input['negcuelabel'].tolist()
    return annotations

def obtain_counts(goldannotations, machineannotations):
    '''
    This function compares the gold annotations to machine output

    :param goldannotations: the gold annotations (list)
    :param machineannotations: the output annotations of the system in question (list)

    :returns: a countainer providing the counts for each predicted and gold class pair as {CLASS1: Counter(CLASS1: x, CLASS2: y)}
    '''
    evaluation_counts = defaultdict(Counter)
    for gold, machine in zip(goldannotations, machineannotations):
        evaluation_counts[gold].update({machine:1})

    return evaluation_counts

def obtain_pr_stats(evaluation_counts):
    ''' This function creates a datastructure containing the True Positives, True Negatives, False Positives and False Negatives
        for each class
        params:
            evaluation_counts: a dict of counters providing the counts for each predicted and gold class pair
        returns:
            stats_dict: a dictonary of defaultdicts as{'TP': {CLASS1: 0, CLASS2:7} ....}

    '''
    stats_dict = {'TP' : defaultdict(int), 'TN' : defaultdict(int), 'FP':defaultdict(int), 'FN' : defaultdict(int)}
    for key, value in evaluation_counts.items():
        for nestedkey, nestedvalue in value.items():
            if key==nestedkey:
                stats_dict['TP'][key] += nestedvalue ## if gold, predicted pair is equal add count as true positive
                for key2 in evaluation_counts.keys():
                    if key2 != key:
                        stats_dict['TN'][key2] += nestedvalue # for each other class, add count as true negatives
            else:
                stats_dict['FP'][nestedkey] += nestedvalue # if gold, predicted keys are not equal, the count is added as false positive
                stats_dict['FN'][key] += nestedvalue # and we missed this count for the key, so add as false negative
    return stats_dict

def calculate_precision_recall_fscore(evaluation_counts):
    '''
    Calculate precision recall and fscore for each class and return them in a dictionary

    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class (dict)

    :returns the precision, recall and f-score of each class (dict)
    '''
    gold_classes = list(evaluation_counts.keys())
    stats_dict = obtain_pr_stats(evaluation_counts)

    report = defaultdict(dict)
    for c in gold_classes:
        TP_c = stats_dict['TP'][c]
        TN_c = stats_dict['TN'][c]
        FP_c = stats_dict['FP'][c]
        FN_c = stats_dict['FN'][c]

        recall_c = TP_c/(TP_c+FN_c)
        try:
            precision_c = TP_c/(TP_c+FP_c)
            f1_c = (2 * precision_c * recall_c) / (precision_c + recall_c)
        except ZeroDivisionError: # if precision is 0, f1 is also 0
            precision_c = 0
            f1_c = 0
        report[c]['precision'] = precision_c
        report[c]['recall'] = recall_c
        report[c]['f-score'] = f1_c
    for unit in ['precision', 'recall', 'f-score']:
        report['avg'][unit] = sum([report[c][unit] for c in gold_classes])/len(gold_classes)
    return report

def provide_confusion_matrix(evaluation_counts):
    '''
    Read in the evaluation counts and provide a confusion matrix for each class

    :param evaluation_counts: a dict containing the true positives, false positives and false negatives for each class

    :prints out a confusion matrix
    '''
    gold_classes = list(evaluation_counts.keys())
    print('\nCONFUSION MATRIX\n')
    all_counts = []
    for c in gold_classes:
        counts = [evaluation_counts[c][cla] for cla in gold_classes]
        all_counts.append(counts)
    df = pd.DataFrame.from_records(all_counts, columns=gold_classes, index=gold_classes)
    print(df.to_latex())


def provide_output_tables(gold_annotations, system_annotations):
    '''
    Create tables based on the evaluation of various systems

    :param evaluations: the outcome of evaluating one or more systems
    '''
    #https:stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
    report = pd.DataFrame(classification_report(gold_annotations, system_annotations, output_dict = True)).transpose()
    print('Precision, recall and f-score:')
    #print(evaluations_pddf)
    print('\n')
    print(report.to_latex())
    evaluation_counts = obtain_counts(gold_annotations, system_annotations)
    provide_confusion_matrix(evaluation_counts)


def main(my_args=None):
    '''
    Runs the evaluation of a single model (see README for instructions)
    '''
    if my_args is None:
        my_args = sys.argv
    gold = my_args[1]
    folder = my_args[2]

    for modelfile in glob.glob(folder + '*.txt'):
        print('FILE:    ', '-'.join(modelfile.split('\\')[-1:][0].split("_"))[:-4])
        gold_annotations = extract_annotations(gold)
        system_annotations = extract_annotations(modelfile)
        provide_output_tables(gold_annotations, system_annotations)



#my_args =['python', r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\SEM-2012-SharedTask-CD-SCO-simple.v2\SEM-2012-SharedTask-CD-SCO-test-preprocessed.txt', r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\Group3_TMwork\models\testset\\']
if __name__ == '__main__':
    print('EVAL')
    main()

#main(my_args)

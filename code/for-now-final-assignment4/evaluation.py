import sys
import pandas as pd
import os
from collections import defaultdict, Counter
import glob

def extract_annotations(inputfile, annotationcolumn, delimiter='\t'):
    '''
    This function extracts annotations represented in the conll format from a file

    :param inputfile: the path to the conll file
    :param annotationcolumn: the name of the column in which the target annotation is provided
    :param delimiter: optional parameter to overwrite the default delimiter (tab)
    :type inputfile: string
    :type annotationcolumn: string
    :type delimiter: string
    :returns: the annotations as a list
    '''
    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    conll_input = pd.read_csv(inputfile, sep=delimiter)
    #print(conll_input.columns)
    annotations = conll_input[annotationcolumn].tolist()
    return annotations

def obtain_counts(goldannotations, machineannotations):
    '''
    This function compares the gold annotations to machine output

    :param goldannotations: the gold annotations
    :param machineannotations: the output annotations of the system in question
    :type goldannotations: the type of the object created in extract_annotations
    :type machineannotations: the type of the object created in extract_annotations

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

    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts

    :returns the precision, recall and f-score of each class in a container
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

    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts

    :prints out a confusion matrix
    '''
    gold_classes = list(evaluation_counts.keys())
    print('CONFUSION MATRIX')
    print(gold_classes)
    all_counts = []
    for c in gold_classes:
        counts = [evaluation_counts[c][cla] for cla in gold_classes]
        all_counts.append(counts)
    df = pd.DataFrame.from_records(all_counts, columns=gold_classes, index=gold_classes)
    print(df.to_latex())

def carry_out_evaluation(gold_annotations, systemfile, systemcolumn, delimiter='\t'):
    '''
    Carries out the evaluation process (from input file to calculating relevant scores)

    :param gold_annotations: list of gold annotations
    :param systemfile: path to file with system output
    :param systemcolumn: indication of column with relevant information
    :param delimiter: specification of formatting of file (default delimiter set to '\t')

    returns evaluation information for this specific system
    '''
    gold_classes = list(set(gold_annotations))
    system_annotations = extract_annotations(systemfile, systemcolumn, delimiter)
    evaluation_counts = obtain_counts(gold_annotations, system_annotations)
    provide_confusion_matrix(evaluation_counts)
    evaluation_outcome = calculate_precision_recall_fscore(evaluation_counts)

    return evaluation_outcome

def provide_output_tables(evaluations):
    '''
    Create tables based on the evaluation of various systems

    :param evaluations: the outcome of evaluating one or more systems
    '''
    #https:stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
    evaluations_pddf = pd.DataFrame.from_dict({(i,j): evaluations[i][j]
                                              for i in evaluations.keys()
                                              for j in evaluations[i].keys()},
                                             orient='index')
    print('Precision, recall and f-score:')
    #print(evaluations_pddf)
    print('\n')
    print(evaluations_pddf.to_latex())

def run_evaluations(goldfile, goldcolumn, systems):
    '''
    Carry out standard evaluation for one or more system outputs

    :param goldfile: path to file with goldstandard
    :param goldcolumn: indicator of column in gold file where gold labels can be found
    :param systems: required information to find and process system output
    :type goldfile: string
    :type goldcolumn: integer
    :type systems: list (providing file name, information on tab with system output and system name for each element)

    :returns the evaluations for all systems
    '''
    evaluations = {}
    #not specifying delimiters here, since it corresponds to the default ('\t')
    gold_annotations = extract_annotations(goldfile, goldcolumn)
    for system in systems:
        sys_evaluation = carry_out_evaluation(gold_annotations, system[0], system[1])
        evaluations[system[2]] = sys_evaluation
    return evaluations

def identify_evaluation_value(system, class_label, value_name, evaluations):
    '''
    Return the outcome of a specific value of the evaluation

    :param system: the name of the system
    :param class_label: the name of the class for which the value should be returned
    :param value_name: the name of the score that is returned
    :param evaluations: the overview of evaluations

    :returns the requested value
    '''
    return evaluations[system][class_label][value_name]

def create_system_information(system_information):
    '''
    Takes system information in the form that it is passed on through sys.argv or via a settingsfile
    and returns a list of elements specifying all the needed information on each system output file to carry out the evaluation.

    :param system_information is the input as from a commandline or an input file
    '''
    # https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    systems_list = [system_information[i:i + 3] for i in range(0, len(system_information), 3)]
    return systems_list

def main(my_args=None):
    '''
    Runs the evaluation of a single model (see README for instructions)
    '''
    if my_args is None:
        my_args = sys.argv

    print('Computing Precision, Recall and F score and confusion matrix for the following parameters:')
    #print('Gold file', my_args[1])
    #print('Model output folder', my_args[3])
    #print('System name as given:', my_args[5])
    print()
    for modelfile in glob.glob(my_args[3] + '*.txt'):
        print(modelfile)
        system_info = create_system_information([modelfile] + my_args[4:])
        #print(system_info)
        evaluations = run_evaluations(my_args[1], my_args[2], system_info)
        print('\n')
        #print(feats)
        provide_output_tables(evaluations)
        #check_eval = identify_evaluation_value('system1', 'O', 'f-score', evaluations)
        #if it does not work correctly, this assert statement will indicate that
        #assert_equal("%.3f" % check_eval,"0.889")


# these can come from the commandline using sys.argv for instance
#os.chdir(r'C:\Users\Tessel Wisman\Documents\TextMining\ml4nlp\ma-ml4nlp-labs\code\assignment3')
my_args =['python', r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\SEM-2012-SharedTask-CD-SCO-simple.v2\SEM-2012-SharedTask-CD-SCO-dev-simple.v2-preprocessed.txt', 'negcuelabel', r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\Group3_TMWork\models\CRF-finalmodels\\', 'negcuelabel', 'SVM']
#if __name__ == '__main__':
main(my_args)

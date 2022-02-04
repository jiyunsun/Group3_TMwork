import sys
import pandas as pd
import os
from evaluation import *
import glob

def compare_incorrect_annotations(gold_file, sys_file, sys_col, outpath, category=None, delimiter='\t'):
    ''' This functions prints the gold data and the system prediction side to side
        in order to examine samples of wrong system predictions
        params:
            gold_file: the path to the gold file (str)
            sys_file: the path to the system output file (str)
            sys_col: the annotation column for the system output file
            outpath: the path of the document that we want to create
            category: optional parameter to print output for only one gold class
            delimiter: delimiter in csv file'''

    gold = pd.read_csv(gold_file, sep=delimiter)

    system_annotations = extract_annotations(sys_file)
    with open(outpath, 'w') as outfile:
        header =['token', 'lemma', 'prev', 'next','pos', 'punct', 'prefix', 'suffix', 'infix', 'sbs_count', 'match_one', 'match_multi', 'prev_tok_cue', 'negcuelabel', 'predictedlabel\n']
        outfile.write('\t'.join(header)) # write the header
        for gold_row, sys in zip(gold.iterrows(), system_annotations):
            if gold_row[1]['negcuelabel'] != sys: # if the prediction does not match the gold label
                gold_row_str= '\t'.join([str(x) for x in list(gold_row[1])]) # convert dataframe to tab separated string
                outfile.write(gold_row_str + '\t' + sys + '\n') # write gold row, prediction

def main(my_args=None):
    '''Run the sample analysis'''
    if my_args is None:
        my_args = sys.argv
    gold = my_args[1]
    system = my_args[2]
    outfile = my_args[3]
    sys_col = 'negcuelabel'
    compare_incorrect_annotations(gold, system, sys_col, outfile)

#args = ['python', '..\SEM-2012-corpus\SEM-2012-dev-preprocessed.txt','..\models\\error-analysis310122_SVMtoken_lemma_prev_next_pos_punct_prefix_suffix_infix_sbs_count_match_one_match_multi_prev_tok_cue.txt', '..\models\\error_analysis-dev.txt']
#main(args)

if __name__ == '__main__':
     main()

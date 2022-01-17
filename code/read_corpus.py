import csv
import sys

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

def get_token(row):
    return row[3]

def get_negcue_label(row):
    return row[4]

def main(args=None):
    if not args:
        args = sys.argv
    path = args[1]
    conll = read_in_conll_file(path)

args = ['x', r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\bioscope-corpus\bioscope.clinical.columns.txt']
main(args)

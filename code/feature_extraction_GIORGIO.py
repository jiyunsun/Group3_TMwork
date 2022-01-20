import csv

#######################################################################################################################################

def read_in_conll_file_for_features(conll_file: str, delimiter: str = '\t'):
    '''
    Read in conll file and return structured object
   
    :param conll_file: path to conll_file
    :param delimiter: specifies how columns are separated. Tabs are standard in conll
   
    :returns List of splitted rows included in conll file
    '''
    my_conll = open(conll_file, 'r')
    conll_as_csvreader = csv.reader(my_conll, delimiter=delimiter, quotechar = delimiter)
    rows = []
    for row in conll_as_csvreader: #skip the blanklines
        if row != []:
            rows.append(row)
    return rows


def extract_all_features(rows, conll_file): #we will create a new conll table, in which we will include all the features
    """
    Creates a conll file with all the features to be represented by one-hot encodings.
    
    rows: the output of the read_in_conll_file function.
    conll_file: the file in which we want to write all the features.
    """
    with open(conll_file, 'w') as output_file:
        for number, row in enumerate(rows):
            
            #TOKEN ITSELF:
            token = row[3]
            token_low = token.lower()
            
            #PREVIOUS TOKEN:
            prev_token = rows[number - 1][3].lower()
            
            #FOLLOWING TOKEN:
            try: 
                next_token = rows[number + 1][3].lower()
            except IndexError:
                next_token = rows[1][3].lower()
            
            #CONTAINS NEGATION INFIX:
            if token_low.endswith("ness"): #instances like "carelessness"
                cut_token_1 = token_low[:-4]
                if cut_token_1.endswith("less"):
                    infix = '1'
                else:
                    infix = '0'
            elif token_low.endswith("ly"): #instances like "restlessly"
                cut_token_2 = token_low[:-2]
                if cut_token_2.endswith("less"):
                    infix = '1'
                else:
                    infix = '0'
            else:
                infix = '0'
            
            #MATCHES ONE-WORD NEGATION EXPRESSION:
            negations = ["no", "not", "nor", "neither", "non", "without", "never", "cannot", "n't", "none", "nothing", "nobody",
                        "nowhere"]
            if token_low in negations:
                onenegexp = '1'
            else:
                onenegexp = '0'

            #MATCHES MULTI-WORD NEGATION EXPRESSION:
            multi_negations = ["no longer", "by no means", "not for the world", "on the contrary", "rather than",
                               "nothing at all", "no more"]
            
            bigram = list() #create bi-grams and check if they match a multi_negation
            bigram.append(token_low)
            try:
                bigram.append(rows[number + 1][3].lower())
            except IndexError:
                bigram.append("")
            biexpression = " ".join(bigram)
            if biexpression in multi_negations:
                multinegexp = '1'
            else:
                reversed_bigram = list()
                reversed_bigram.append(rows[number - 1][3].lower())
                reversed_bigram.append(token_low)
                reversed_biexpression = " ".join(reversed_bigram)
                if reversed_biexpression in multi_negations:
                    multinegexp = '1'
                else:
                    trigram = list() #create tri-grams and check if they match the multi_negations
                    trigram.append(token_low)
                    try:
                        trigram.append(rows[number + 1][3].lower())
                    except IndexError:
                        trigram.append("")
                    try:
                        trigram.append(rows[number + 2][3].lower())
                    except IndexError:
                        trigram.append("")
                    triexpression = " ".join(trigram)
                    if triexpression in multi_negations:
                        multinegexp = '1'
                    else:
                        centered_trigram = list()
                        centered_trigram.append(rows[number - 1][3].lower())
                        centered_trigram.append(token_low)
                        try:
                            centered_trigram.append(rows[number + 1][3].lower())
                        except IndexError:
                            centered_trigram.append("")
                        centered_triexpression = " ".join(centered_trigram)
                        if centered_triexpression in multi_negations:
                            multinegexp = '1'
                        else:
                            reversed_trigram = list()
                            reversed_trigram.append(rows[number - 2][3].lower())
                            reversed_trigram.append(rows[number - 1][3].lower())
                            reversed_trigram.append(token_low)
                            reversed_triexpression = " ".join(reversed_trigram)
                            if reversed_triexpression in multi_negations:
                                multinegexp = '1'
                            else:
                                fourgram = list() #create four-grams and check if they match the multi_negations
                                fourgram.append(token_low)
                                try:
                                    fourgram.append(rows[number + 1][3].lower())
                                except IndexError:
                                    fourgram.append("")
                                try:
                                    fourgram.append(rows[number + 2][3].lower())
                                except IndexError:
                                    fourgram.append("")
                                try: 
                                    fourgram.append(rows[number + 3][3].lower())
                                except IndexError:
                                    fourgram.append("")
                                fourexpression = " ".join(fourgram)
                                if fourexpression in multi_negations:
                                    multinegexp = '1'
                                else:
                                    centeredup_fourgram = list()
                                    centeredup_fourgram.append(rows[number - 1][3].lower())
                                    centeredup_fourgram.append(token_low)
                                    try:
                                        centeredup_fourgram.append(rows[number + 1][3].lower())
                                    except IndexError:
                                        centeredup_fourgram.append("")
                                    try: 
                                        centeredup_fourgram.append(rows[number + 2][3].lower())
                                    except IndexError:
                                        centeredup_fourgram.append("")
                                    centeredup_fourexpression = " ".join(centeredup_fourgram)
                                    if centeredup_fourexpression in multi_negations:
                                        multinegexp = '1'
                                    else:
                                        centereddown_fourgram = list()
                                        centereddown_fourgram.append(rows[number - 2][3].lower())
                                        centereddown_fourgram.append(rows[number - 1][3].lower())
                                        centereddown_fourgram.append(token_low)
                                        try:
                                            centereddown_fourgram.append(rows[number + 1][3].lower())
                                        except IndexError:
                                            centereddown_fourgram.append("")
                                        centereddown_fourexpression = " ".join(centereddown_fourgram)
                                        if centereddown_fourexpression in multi_negations:
                                            multinegexp = '1'
                                        else:
                                            reversed_fourgram = list()
                                            reversed_fourgram.append(rows[number - 3][3].lower())
                                            reversed_fourgram.append(rows[number - 2][3].lower())
                                            reversed_fourgram.append(rows[number - 1][3].lower())
                                            reversed_fourgram.append(token_low)
                                            reversed_fourexpression = " ".join(reversed_fourgram)
                                            if reversed_fourexpression in multi_negations:
                                                multinegexp = '1'
                                            else:
                                                multinegexp = '0'
                                
            features_list = [token_low, prev_token, next_token, infix, onenegexp, multinegexp]
            output_file.write('\t'.join(features_list) + "\n")

            
def extract_embeddings_as_features_and_gold(conllfile, word_embedding_model):
    '''
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
        if len(row) > 3: #check for cases where empty lines mark sentence boundaries (which some conll files do).
            if row[3] in word_embedding_model: #TESSEL SHOULD WE ADD .lower() TO THE TOKEN?
                vector = word_embedding_model[row[3]]
            else:
                vector = [0]*300 #TESSEL I'M NOT SURE ABOUT THAT, SHOULD BE [3] TOO?
            features.append(vector)
            labels.append(row[-1])
    return features, labels
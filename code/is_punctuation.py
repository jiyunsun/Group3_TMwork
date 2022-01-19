import string

def is_punctuation(row):
      
    for token in row[3]:
        if token in string.punctuation:
            print(int(1))
        else: 
            print(int(0))
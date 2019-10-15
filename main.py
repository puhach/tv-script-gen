import os
import torch
import checkpoint
#import problem_unittests as tests
from collections import Counter

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    
    word_counter = Counter(text)
    #print(word_counter)
    vocab = sorted(word_counter, key=word_counter.get, reverse=True)
    #print(vocab)
    
    vocab_to_int = { word: i for i, word in enumerate(vocab) }
    int_to_vocab = { i:word for word,i in vocab_to_int.items() }
    
    #print(vocab_to_int)
    #print(int_to_vocab)
    
    # return tuple
    return (vocab_to_int, int_to_vocab)


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    punctuation_map = { '.':'||period||', ',':'||comma||', '"':'||quote||', ';':'||semicolon||',
                      '!':'||exclamation||', '?':'||question||', '(':'||leftpar||', ')':'||rightpar||',
                      '-':'||dash||', '\n':'||return||' }    
    
        
    return punctuation_map

try:
    _, vocab_to_int, int_to_vocab, token_dict = checkpoint.load_preprocess()
    trained_rnn = checkpoint.load_model('./save/trained_rnn')
except:
    data_dir = './data/Seinfeld_Scripts.txt'
    text = checkpoint.load_data(data_dir)


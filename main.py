import torch
import checkpoint
#import problem_unittests as tests

try:
    _, vocab_to_int, int_to_vocab, token_dict = checkpoint.load_preprocess()
    trained_rnn = checkpoint.load_model('./save/trained_rnn')
except:
    data_dir = './data/Seinfeld_Scripts.txt'
    text = load_data(data_dir)
    
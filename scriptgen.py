import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import preprocessor
from rnn import RNN
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    
    word_counter = Counter(text)
    vocab = sorted(word_counter, key=word_counter.get, reverse=True)
    
    vocab_to_int = { word: i for i, word in enumerate(vocab) }
    int_to_vocab = { i:word for word,i in vocab_to_int.items() }
    
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


def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    n_sequences = len(words)//sequence_length
    features = np.zeros(shape=(n_sequences, sequence_length), dtype=np.int64)
    targets = np.zeros(shape=(n_sequences), dtype=np.int64)
    
    for i in range(0, n_sequences):
        start = i*sequence_length        
        features[i, :] = words[start: start+sequence_length]
        if start+sequence_length >= len(words):
            targets[i] = words[0]
        else:
            targets[i] = words[start+sequence_length]
        
   
    data_tensor = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))
    data_loader = DataLoader(data_tensor, batch_size=batch_size, shuffle=True)
    
    return data_loader


def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden, use_gpu):
    """
    Forward and backward propagation on the neural network
    :param rnn: The PyTorch Module that holds the neural network
    :param optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :param hidden: The initial hidden state
    :param use_gpu: Determines whether to use GPU for accelerated training
    :return: The loss and the latest hidden state Tensor
    """
        
    # move data to GPU, if available
    if use_gpu:
        inp = inp.cuda()
        target = target.cuda()
    
    # perform backpropagation and optimization
    out, hidden = rnn(inp, hidden)
    optimizer.zero_grad()
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()

    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), hidden


def train_rnn(rnn, train_loader, batch_size, optimizer, criterion, n_epochs, use_gpu, show_every_n_batches=100):
    """
    Trains the model on the preprocessed data.
    :param rnn: The recurrent neural network to be trained
    :param train_loader: Provides an iterable over the training set
    :param batch_size: The number of word sequences in a batch
    :param optimizer: The optimizer for updating model's weights
    :param criterion: The loss function
    :param n_epochs: The number of iterations over the entire training set
    :param use_gpu: Determines whether GPU should be used to accelerate training
    :show_every_n_batches: Prints loss statistics when the specified number of batches have been processed
    :return: The trained model
    """

    batch_losses = []
    
    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size, use_gpu)
        
        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            
            # make sure we iterate over completely full batches, only
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
            
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden, use_gpu)          
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []

    # returns a trained rnn
    return rnn


def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, sequence_length, predict_len, use_gpu):
    """
    Generate text using the neural network
    :param decoder: The PyTorch Module that holds the trained neural network
    :param prime_id: The word id to start the first prediction
    :param int_to_vocab: Dict of word id keys to word values
    :param token_dict: Dict of puncuation tokens keys to puncuation values
    :param pad_value: The value used to pad a sequence
    :param sequence_len: The number of words in a sequence the model was trained for
    :param predict_len: The length of text to generate
    :param use_gpu: Determines whether GPU should be used to speed up calculations
    :return: The generated text
    """
    rnn.eval()
    
    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, sequence_length), pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]
    
    for _ in range(predict_len):
        if use_gpu:
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)
        
        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0), use_gpu)
        
        # get the output of the rnn
        output, _ = rnn(current_seq, hidden)
        
        # get the next word probabilities
        p = nn.functional.softmax(output, dim=1).data
        if use_gpu:
            p = p.cpu() # move to cpu
         
        # use top_k sampling to get the index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()
        
        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p/p.sum())
        
        # retrieve that word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)     
        
        # the generated word becomes the next "current sequence" and the cycle can continue
        current_seq = np.roll(current_seq, -1, 1)
        current_seq[-1][-1] = word_i
    
    gen_sentences = ' '.join(predicted)
    
    # Replace punctuation tokens
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)

    gen_sentences = gen_sentences.replace('\n ', '\n')
    gen_sentences = gen_sentences.replace('( ', '(')
    
    # return all the sentences
    return gen_sentences

    



parser = argparse.ArgumentParser(description='TV Script Generator')
parser.add_argument('--prime-word', type=str, required=True,
                    help='the first word of a new script (the name of a character)')
parser.add_argument('--script-len', type=int, required=True,
                    help='the length of a script to generate')
parser.add_argument('--n-epochs', type=int, default=2,
                    help='the number of epochs to train the model for')
parser.add_argument('--learning-rate', type=float, default=0.0001,
                    help='the learning rate for gradient descent')
parser.add_argument('--sequence-len', type=int, default=16,
                    help='the number of words in a sequence')
parser.add_argument('--batch-size', type=int, default=32,
                    help='the number of word sequences in a batch')
parser.add_argument('--n-layers', type=int, default=2, 
                    help='the number of layers in the recurrent neural network')
parser.add_argument('--embedding-dim', type=int, default=256,
                    help='the embedding dimension')
parser.add_argument('--hidden-dim', type=int, default=512,
                    help='the hidden dimension of the recurrent neural network')
parser.add_argument('--stat-freq', type=int, default=200, 
                    help='prints loss statistics after the specified number of batches have been processed')


args = parser.parse_args()
print(args)

use_gpu = torch.cuda.is_available()
if not use_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')

# Number of words in a sequence.
# This parameter is used both for training and generating.
sequence_length = args.sequence_len 

try:
    print("Loading the model...")
    _, vocab_to_int, int_to_vocab, token_dict = preprocessor.load_preprocess()
    trained_rnn = preprocessor.load_model('./save/trained_rnn')

except:
    print("Unable to load a checkpoint. Input data needs to be preprocessed.")

    data_dir = './data/Seinfeld_Scripts.txt'
    text = preprocessor.load_data(data_dir)

    int_text, vocab_to_int, int_to_vocab, token_dict = preprocessor.preprocess_and_save_data(data_dir, 
        token_lookup, create_lookup_tables)

    
    batch_size = args.batch_size # 32

    train_loader = batch_data(int_text, sequence_length, batch_size)

    print("Training...")
    
    #num_epochs = 20
    num_epochs = args.n_epochs
    learning_rate = args.learning_rate #0.0001
    vocab_size = len(vocab_to_int)
    output_size = vocab_size
    embedding_dim = args.embedding_dim #256
    hidden_dim = args.hidden_dim #512
    n_layers = args.n_layers #2   
    show_every_n_batches = args.stat_freq # Show stats for every n number of batches

    rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
    if use_gpu:
        rnn.cuda()

    # defining loss and optimization functions for training
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # training the model
    trained_rnn = train_rnn(rnn, 
        train_loader, 
        batch_size, 
        optimizer, 
        criterion, 
        num_epochs, 
        use_gpu, 
        show_every_n_batches)

    # saving the trained model
    preprocessor.save_model('./save/trained_rnn', trained_rnn)


print("Generating a script...")

gen_length = args.script_len # modify the length to your preference
prime_word = args.prime_word.lower() # elaine

if not prime_word+':' in vocab_to_int.keys():
    print('Failed to generate a script (wrong prime word).')
    sys.exit()

pad_word = preprocessor.SPECIAL_WORDS['PADDING']

generated_script = generate(trained_rnn, 
    vocab_to_int[prime_word + ':'], 
    int_to_vocab, 
    token_dict, 
    vocab_to_int[pad_word], 
    sequence_length,
    gen_length,
    use_gpu)

print(generated_script)
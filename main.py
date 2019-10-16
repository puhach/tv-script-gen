import os
import numpy as np
import torch
import torch.nn as nn
import checkpoint
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
    features = np.zeros(shape=(n_sequences, sequence_length), dtype=np.int)
    targets = np.zeros(shape=(n_sequences), dtype=np.int)
    
    for i in range(0, n_sequences):
        start = i*sequence_length        
        features[i, :] = words[start: start+sequence_length]
        if start+sequence_length >= len(words):
            targets[i] = words[0]
        else:
            targets[i] = words[start+sequence_length]
        
   
    # return a dataloader
    #data_tensor = TensorDataset(features, targets)
    data_tensor = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))
    data_loader = DataLoader(data_tensor, batch_size=batch_size, shuffle=True)
    
    #x,y = next(iter(data_loader))
    #print(type(x))
    #for i in range(x.shape[0]):
    #    found = False
    #    for j in range(features.shape[0]):
    #        eq = x[i].numpy() == features[j]            
    #        if eq.sum() == x.shape[1] and y[i].item() == targets[j]:
    #            found = True
    #            break
    #    print("found:", found)
    
    return data_loader


def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden, train_on_gpu):
    """
    Forward and backward propagation on the neural network
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
        
    # move data to GPU, if available
    if train_on_gpu:
        inp = inp.cuda()
        target = target.cuda()
    
    # perform backpropagation and optimization
    #print("inp:", inp)
    out, hidden = rnn(inp, hidden)
    optimizer.zero_grad()
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()

    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), hidden


def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, train_on_gpu, show_every_n_batches=100):
    batch_losses = []
    
    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size, train_on_gpu)
        
        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            
            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
            
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden, train_on_gpu)          
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []

    # returns a trained rnn
    return rnn



try:
    _, vocab_to_int, int_to_vocab, token_dict = checkpoint.load_preprocess()
    trained_rnn = checkpoint.load_model('./save/trained_rnn')

except:
    data_dir = './data/Seinfeld_Scripts.txt'
    text = checkpoint.load_data(data_dir)

    int_text, vocab_to_int, int_to_vocab, token_dict = checkpoint.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

    sequence_length = 16  # number of words in a sequence
    batch_size = 32

    train_loader = batch_data(int_text, sequence_length, batch_size)

    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('No GPU found. Please use a GPU to train your neural network.')
    
    num_epochs = 20
    learning_rate = 0.0001
    vocab_size = len(vocab_to_int)
    output_size = vocab_size
    embedding_dim = 256
    hidden_dim = 512
    n_layers = 2

    # Show stats for every n number of batches
    show_every_n_batches = 200

    rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
    if train_on_gpu:
        rnn.cuda()

    # defining loss and optimization functions for training
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # training the model
    trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, train_on_gpu, show_every_n_batches)

    # saving the trained model
    checkpoint.save_model('./save/trained_rnn', trained_rnn)
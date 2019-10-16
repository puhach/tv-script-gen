import torch.nn as nn

class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        
        # set class variables
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.hidden_size = hidden_dim
        self.n_layers = n_layers
        #self.train_on_gpu = train_on_gpu
        
        # define model layers
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, 
                          num_layers=n_layers, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_size)
        
    
    
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        # TODO: Implement function   
        #print("input:", nn_input.shape) 
        x = self.emb(nn_input) # (batch_size, seq_len) -> (batch_size, seq_len, emb_dim)
        #print("emb:", x.shape)
        
        ## ! Please, update the tests. GRU's state is not a tuple !
        #if isinstance(hidden, tuple):
        #    hidden = hidden[0]
            
        x, h = self.gru(x, hidden) # (batch_size, seq_len, emb_dim) -> (batch_size, seq_len, hidden_size)
        #print("gru:", x.shape)
        
        # detach the hidden state from the computation graph to prevent it from propagating back 
        # throughout the whole time sequence
        h = h.detach()
        
        # as long as we are concerned about only the last batch, we can drop the rest of them here
        # without even passing to the fully connected layer
        x = x[:, -1, :] # (batch_size, hidden_size)
        
        # randomly disable some neurons to prevent overfitting
        x = self.drop(x)

        # adjust the output to our desired format, i.e.
        # the number of features will be resized to match the output size
        x = self.fc(x)
        
        # return one batch of output word scores and the hidden state
        ## ! Again, no tuple is needed for the hidden state, but the test function
        ## will not accept it otherwise... !
        #return x, (h,)
        return x, h
    
    
    def init_hidden(self, batch_size, train_on_gpu):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        
        # ! According to the documentation, if the initial hidden state is not provided, 
        # it defaults to zero, hence I don't see the point to manually initialize it with 
        # zero weights !
                
        # initialize hidden state with zero weights, and move to GPU if available
        weight = next(self.parameters()).data
        if train_on_gpu: # num_layers * num_directions, batch_size, hidden_size
            hidden = weight.new(self.n_layers, batch_size, self.hidden_size).zero_().cuda()
        else:
            hidden = weight.new(self.n_layers, batch_size, self.hidden_size).zero_()
        
        #print("hidden:", hidden.shape)
        
        ## ! in contrast with LSTM, GRU's hidden state is not a tuple, but the test function
        ## seems to expect a tuple, so I have to pack it this way !
        #return (hidden,)
        return hidden
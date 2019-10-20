# TV Script Generator

This project demonstrates the use of RNNs for script generation. It will create a new "fake" TV script based on the 9 seasons of Seinfeld TV serries. 

## Setup

Install PyTorch (I used v1.2.0 for development):
```
pip install torch
```

Clone this repo:
```
git clone https://github.com/puhach/tv-script-gen.git
```

## Usage

To start generating a new script specify the first word (should be the name of a character) and the script length:
```
scriptgen.py --prime-word elaine --script-len 333
```

Unless the model has already been trained, it will start the training process which make take a long time. 
Once training is finished, model's checkpoint is saved to 'train_rnn.pt' file and the script will be generated.


Further script generation will be using the pre-trained model, so the script should be printed shortly. 
In order to retrain the model, delete the 'trained_rnn.pt' file and launch generation with a different set of parameters.

Parameter    | Meaning | Example
------------ | ------------- | -------------------------
prime-word | [Required] The first word of a new script (the name of a character).  | --prime-word jerry
script-len | [Required] The length of a script to generate. | --script-len 123
n-epochs | [Optional] The number of epochs to train the model for. Default is 2. | --n-epochs 10
learning-rate | [Optional] The learning rate for gradient descent. Default is 0.0001. | --learning-rate 0.01
sequence-len | [Optional] The number of words in a sequence. Default is 16. | --sequence-len 16
batch-size | [Optional] The number of word sequences in a batch. Default is 32. | --batch-size 32
n-layers | [Optional] The number of layers in the recurrent neural network. Default is 2. | --n-layers 2
embedding-dim | [Optional] The embedding dimension. Default is 256. | --embedding-dim 128
hidden-dim | [Optional] The hidden dimension of the recurrent neural network. Default is 512. | --hidden-dim 256
stat-freq | [Optional] Prints loss statistics after the specified number of batches have been processed. Default is 200. | --stat-freq 100



## Acknowledgements

The project was implemented as a part of [Udacity Deep Learning nanodegree](https://github.com/udacity/deep-learning-v2-pytorch).

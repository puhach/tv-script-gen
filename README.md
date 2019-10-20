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



## Acknowledgements

The project was implemented as a part of [Udacity Deep Learning nanodegree](https://github.com/udacity/deep-learning-v2-pytorch).

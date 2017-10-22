# spell_correction_LSTM
A sequence-to-sequence model on spell correction using LSTM in keras.

## Introduction

I this projecct, I try to build a encoder-decorder model to correct typo in English. 
Encoder-decorder model is not very difficult to build, the only thing we need to keep in mind is the input and output shape of the model. 
The concept of sequence-to-sequence model can be found in the references I provide. </br>
But the problem is, sequence-to-sequence model takes toooo long to train. 
Even for a simple Echo Sequences of Random Integers, we need to train 5000 rounds to see the correct result. </br>
To deal with text, it takes much larger space, because we need to encode the characters to on-hot vectors.
You may have errors says that you don't have enough memory. </br>
So, it is very very slow to train and takes up a lot of memeory. Try to run the code only if you have a gpu and at least one or two days' time.

## Methodology

1. Prepare the dataset, add some noise to the original dataset. And then vectorize the chars.
2. Build a sequence to sequence model, with LSTMs as encoder and decoder. 
3. Train the model.

## Result

Currently, I'm not able to finish running the model. Because I do not have enough time and another computer to run on. 
I'll try to run it if I have time later.

## Some points to note

1. Why we use RepeatVector layer after encoder LSTM.
> The encoder layer will output a 2D array (, ) and the decoder expects a 3D array as input (, ?, ).
We address this problem by adding a RepeatVector() layer between the encoder and decoder 
and ensure that the output of the encoder is repeated a suitable number of times to match the length of the output sequence.

2. Why we need TimeDistributed layer after decoder LSTM.
> The TimeDistributed layer performs the trick of applying each slice of the sequence from the LSTM layer as inputs 
to the wrapped Dense layer so that one integer can be predicted at a time. 

 3. Why we need to vectorized the characters.
 > Once we have integers, we need to transform them into a format that is suitable for training an LSTM network. 
 One option would be to rescale the integer to the range [0,1]. This would work and would require that the problem be phrased as regression.
 I am interested in predicting the right number, not a number close to the expected value. 
 This means I would prefer to frame the problem as classification rather than regression, 
 where the expected output is a class and there are 100 possible class values.
 
Please see more explanations in the references below. Those are very good articles to read.

## References
https://machinelearningmastery.com/how-to-use-an-encoder-decoder-lstm-to-echo-sequences-of-random-integers/ </br>
https://github.com/surmenok/DeepSpell/blob/master/keras_spell.py </br>
https://machinelearningmastery.com/lstms-with-python/ </br>
https://chunml.github.io/ChunML.github.io/project/Sequence-To-Sequence/

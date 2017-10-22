# -*- coding: utf-8 -*-
import random
import numpy as np
#from sklearn.model_selection import train_test_split


class DataHelper():
    
    def __init__(self):
        
        self.CHARS = list("abcdefghijklmnopqrstuvwxyz .")
        #self.corpus_path = '/home/ec2-user/spell_correction_keras/data/europarl-v8.fi-en.en' #europarl-v8.fi-en.en
        self.corpus_path = '/workspace/spell_correction_keras/data/europarl-v8.fi-en.en'
    
    def add_noise_to_string(self, a_string, amount_of_noise):
        """Add some artificial spelling mistakes to the string"""
        if random.random() < amount_of_noise * len(a_string):
            # Replace a character with a random character
            random_char_position = random.randint(0, len(a_string))
            a_string = a_string[:random_char_position] + random.choice(self.CHARS[:-1]) + a_string[random_char_position + 1:]
        if random.random() < amount_of_noise * len(a_string):
            # Delete a character
            random_char_position = random.randint(0, len(a_string))
            a_string = a_string[:random_char_position] + a_string[random_char_position + 1:]
        if random.random() < amount_of_noise * len(a_string):
            # Add a random character
            random_char_position = random.randint(0, len(a_string))
            a_string = a_string[:random_char_position] + random.choice(self.CHARS[:-1]) + a_string[random_char_position:]
        if random.random() < amount_of_noise * len(a_string):
            # Transpose 2 characters
            random_char_position = random.randint(0, len(a_string) - 2)
            a_string = (a_string[:random_char_position] +
                        a_string[random_char_position + 1] +
                        a_string[random_char_position] +
                        a_string[random_char_position + 2:])
        return a_string
    
    def data_description(self):
        
        f = open(self.corpus_path, 'r')#, encoding = 'utf-8')
        y_raw = f.read().lower()
        f.close()
        
        # Splitting raw text into a list of sentences
        y_sents = y_raw.split('\n')
        
        self.chars = sorted(list(set(list(y_raw) + self.CHARS)))
        print('Unique chars: ', len(self.chars))
        
        self.char_to_int = dict((c, i) for i, c in enumerate(self.chars))
        self.int_to_char = dict((i, c) for i, c in enumerate(self.chars))
        
        self.MAX_OUTPUT_LEN = max(len(sent) for sent in y_sents)
        self.NOISE = 0.4/self.MAX_OUTPUT_LEN
        
        # Add noise to the sentences and inverse the sentence if inverse = True
        x_sents = [self.add_noise_to_string(sent, self.NOISE) for sent in y_sents]      
        self.MAX_INPUT_LEN = max(len(sent) for sent in x_sents)
        
        print('MAX_INPUT_LEN: ', self.MAX_INPUT_LEN, '\nMAX_OUTPUT_LEN: ', self.MAX_OUTPUT_LEN)
        return self.MAX_OUTPUT_LEN, self.MAX_INPUT_LEN, self.chars, self.int_to_char
    
    def load_data(self, epoch, batch_size, inverse = True):
        
        f = open(self.corpus_path, 'r')#, encoding = 'utf-8')
        y_raw = f.read().lower()
        f.close()
        
        # Splitting raw text into a list of sentences
        y_sents = y_raw.split('\n')[epoch* batch_size: epoch* batch_size + batch_size]
        x_sents = [self.add_noise_to_string(sent, self.NOISE)[::-1] if inverse else self.add_noise_to_string(sent, self.NOISE) for sent in y_sents]      

        x = np.zeros((len(x_sents), self.MAX_INPUT_LEN, len(self.chars)), dtype=np.int)
        y = np.zeros((len(y_sents), self.MAX_OUTPUT_LEN, len(self.chars)), dtype=np.int)
        
        for i, sentence in enumerate(x_sents):
            for j, char in enumerate(sentence):
                x[i][j][self.char_to_int[char]] = 1
        
        for i, sentence in enumerate(y_sents):
            for j, char in enumerate(sentence):
                y[i][j][self.char_to_int[char]] = 1
                
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2017)
        
        return x, y #x_train, x_test, y_train, y_test

import sys
#sys.path.insert(0, '/home/ec2-user/spell_correction_keras/code/*')
sys.path.insert(0, '/workspace/spell_correction_keras/code/*')
from prepare_data import DataHelper
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Dropout
from keras.layers import recurrent
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

class TrainModel():
    
    def __init__(self, inverse):
        
        self.inverse = inverse
        self.helper = DataHelper()
        self.INPUT_LAYERS = 2
        self.OUTPUT_LAYERS = 2
        self.AMOUNT_OF_DROPOUT = 0.2
        self.HIDDEN_SIZE = 700
        self.BATCH_SIZE = 20
        self.INITIALIZATION = "he_normal"  # : Gaussian initialization scaled by fan_in (He et al., 2014)
        self.output_len, self.input_len, self.chars, self.int_to_char = self.helper.data_description()
        self.x_test = self.helper.load_data(0, self.BATCH_SIZE, self.inverse)
        

    def create_model(self):
        
        print('Build model...')
        model = Sequential()
        # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
        # note: in a situation where your input sequences have a variable length,
        # use input_shape=(None, nb_feature).
        for layer_number in range(self.INPUT_LAYERS):
            model.add(recurrent.LSTM(self.HIDDEN_SIZE, input_shape=(self.input_len, len(self.chars)), #init=self.INITIALIZATION, input_shape=(self.input_len, len(self.chars))
                                     return_sequences=layer_number + 1 < self.INPUT_LAYERS))
            model.add(Dropout(self.AMOUNT_OF_DROPOUT))
        print('Input: ', model.input_shape)
        print('Encoder output: ', model.output_shape)
        # For the decoder's input, we repeat the encoded input for each time step
        model.add(RepeatVector(self.output_len))
        print('RepeatVector output: ', model.output_shape)
        # The decoder RNN could be multiple layers stacked or a single layer
        for _ in range(self.OUTPUT_LAYERS):
            model.add(recurrent.LSTM(self.HIDDEN_SIZE, return_sequences=True))#, init=self.INITIALIZATION, stateful=True))
            model.add(Dropout(self.AMOUNT_OF_DROPOUT))
    
        # For each of step of the output sequence, decide which character should be chosen
        model.add(TimeDistributed(Dense(len(self.chars), init=self.INITIALIZATION)))
        model.add(Activation('softmax'))
        print('Decoder output: ', model.output_shape)
    
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        return model
    
    def train_model(self, model):
        
        #filepath = '/home/ec2-user/spell_correction_keras/data/weights-improvement-{epoch:02d}-{acc:.4f}.hdf5'
        filepath = '/workspace/spell_correction_keras/data/weights-improvement-{epoch:02d}-{acc:.4f}.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        
        for epoch in range(1, 7001):
            print('------------------- epoch: ', epoch)
            #x_train, x_test, y_train, y_test  = self.helper.load_data(epoch, self.BATCH_SIZE, self.inverse)
            x_train, y_train = self.helper.load_data(epoch, self.BATCH_SIZE, self.inverse)
            model.fit(x_train, y_train, batch_size=self.BATCH_SIZE, epochs=1, verbose=2, callbacks=callbacks_list)
        
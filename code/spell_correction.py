import os
import sys
import numpy as np
#sys.path.insert(0, '/home/ec2-user/spell_correction_keras/code/*')
sys.path.insert(0, '/workspace/spell_correction_keras/code/*')
from train_model import TrainModel

inverse = False

model_helper = TrainModel(inverse)
model = model_helper.create_model()

has_model = sum([True if file.startswith('weight') else False for file in os.listdir('/home/ec2-user/spell_correction_keras/data/')])
if has_model == 0:
    model_helper.train_model(model)
else:
    #filename = '/home/ec2-user/spell_correction_keras/data/weights-improvement-00-1.0991.hdf5' 
    filename = '/workspace/spell_correction_keras/data/weights-improvement-00-1.0991.hdf5' 
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

results =   model.predict(model_helper.x_test)
print(results[0:2])
predictions = [[np.argmax(word) for word in sentence] for sentence in results]
sums = [[sum(word) for word in sentence] for sentence in results]
print('arg data: ', predictions)  
sequences = []
for i in len(predictions):
    prediction = predictions[i]
    if inverse:
        sequence = ''.join([model_helper.int_to_char[index] for index in prediction])[::-1]
    else:
        sequence = ''.join([model_helper.int_to_char[index] for index in prediction])
    print('?????: ', sequence)
    print('+++++: ', model_helper.y_test[i])
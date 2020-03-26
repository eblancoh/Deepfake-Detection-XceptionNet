import keras
import numpy as np
from keras.applications import xception
from keras.models import model_from_json, Model
from keras.layers import Input, Dense, Dropout, LSTM, TimeDistributed

def new_model(image_size = 299):
    inputs = Input(shape=(image_size, image_size, 3))
    model = xception.Xception(include_top=False, weights='imagenet')

    model = Model(inputs=inputs, outputs=model)
    
    sgd= keras.optimizers.SGD(momentum=0.9, lr=0.045, decay=1e-6)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    model.summary()
    return model

def save_model(model,name):
    model_json = model.to_json()
    with open(name + ".json", "w") as json_file:
        json_file.write(model_json)
        
    model.save_weights(name + "_weight.h5")
    print("Saved model to disk")
    
def load_model(name):
    json_file = open(name + '.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    
    model.load_weights(name + "_weight.h5")
    print("Loaded model from disk")
 
    adam = keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(loss='binary_crossentropy', optimizer = adam, 
                  metrics=['accuracy'])
    print("Model compiled")
    return model
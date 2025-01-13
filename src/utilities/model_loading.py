from keras_facenet import FaceNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import json

def load_facenet_model(label_map_path, model_path):
    with open(label_map_path, 'r') as json_file:
        labels = list(json.load(json_file).keys())
        num_classes = len(labels)
    
    facenet = FaceNet()
    base_model = facenet.model
    
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output_layer)
    
    model.load_weights(model_path)
    return model

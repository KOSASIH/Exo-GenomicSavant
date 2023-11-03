from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def create_rnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=128, input_length=input_shape[1]))
    model.add(LSTM(64))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

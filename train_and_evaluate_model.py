from tensorflow.keras.callbacks import EarlyStopping

def train_and_evaluate_model(model, train_sequences, train_labels, test_sequences, test_labels):
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)
    
    model.fit(train_sequences, train_labels, validation_data=(test_sequences, test_labels),
              epochs=10, batch_size=32, callbacks=[early_stopping])
    
    _, accuracy = model.evaluate(test_sequences, test_labels)
    
    return accuracy

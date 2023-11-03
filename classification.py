# Preprocessing the dataset
dataset_path = 'path/to/dataset'
max_sequence_length = 1000
train_sequences, test_sequences, train_labels, test_labels, label_mapping = preprocess_dataset(dataset_path, max_sequence_length)

# Creating and training the CNN model
cnn_model = create_cnn_model(train_sequences.shape, len(label_mapping))
cnn_accuracy = train_and_evaluate_model(cnn_model, train_sequences, train_labels, test_sequences, test_labels)

# Creating and training the RNN model
rnn_model = create_rnn_model(train_sequences.shape, len(label_mapping))
rnn_accuracy = train_and_evaluate_model(rnn_model, train_sequences, train_labels, test_sequences, test_labels)

print(f"CNN Accuracy: {cnn_accuracy}")
print(f"RNN Accuracy: {rnn_accuracy}")

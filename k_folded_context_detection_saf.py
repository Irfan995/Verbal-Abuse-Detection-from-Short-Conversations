import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from contextlib import redirect_stdout
from sklearn.model_selection import KFold

# Configure logging to file for tracking training progress and metrics
logging.basicConfig(filename='dynamic_dataset_logs/violence_detection_k_folded.log', level=logging.INFO)
logger = logging.getLogger()

# Custom callback to log epoch metrics using the Python logger
class ModelLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logger.info(f'Epoch {epoch+1}: {logs}')

def calculate_layer_input(dimension):
    """
    Calculate sizes for various network components based on a base embedding dimension.
    Returns:
      filter: number of Conv1D filters
      input_layer: base input dimension (same as embedding dim)
      hidden_layer_one: size for first dense hidden layer
      hidden_layer_two: size for second dense hidden layer
    """
    filter = (dimension * 2) / 4
    input_layer = dimension
    hidden_layer_one = (dimension * 2) / 3
    hidden_layer_two = dimension / 3
    return int(filter), int(input_layer), int(hidden_layer_one), int(hidden_layer_two)

# Determine layer sizes from the SBERT embedding dimension (768 for this model)
filter, input_layer, hidden_layer_one, hidden_layer_two = calculate_layer_input(768)

# Load dataset from Excel file and keep only the relevant columns
df = pd.read_excel(r"domestic_violence_dataset.xlsx")
df = df[['Text', 'Flag_type']] # LABEL_COLUMNS

# Normalize text to lowercase
df['Text'] = df['Text'].str.lower()

# Encode labels with LabelBinarizer for multi-class / multi-label usage
multi_label = LabelBinarizer()
labels = multi_label.fit_transform(df['Flag_type'])
total_classes = len(multi_label.classes_)  # number of unique label classes

# Convert label matrix to a DataFrame of float32 (useful for Keras training)
multi_label_df = pd.DataFrame(labels, columns=multi_label.classes_)
# Note: the next line performs a conversion but the resulting array is not assigned back;
# the DataFrame multi_label_df will still be used downstream.
np.asarray(multi_label_df).astype('float32').reshape((-1,1))

# Encode text samples to numeric vectors using a SentenceTransformer
voice_list = df['Text'].to_numpy()
sbert = SentenceTransformer('all-distilroberta-v1')
sentence_embeddings = sbert.encode(voice_list)

# Keep tokenizer and its vocab for use inside the model (Embedding layer requires vocab size)
tokenizer = sbert.tokenizer
vocab = tokenizer.vocab

# Split dataset into fixed train and test sets (stratified by original label)
train_x, test_x, train_y, test_y = train_test_split(
    sentence_embeddings,
    multi_label_df,
    train_size=0.7,
    test_size=0.3,
    random_state=42,
    stratify=df['Flag_type']
)

def create_model(input_layer, hidden_layer_one, hidden_layer_two, total_classes):
    """
    Build and compile a Keras model that mixes an attention-style block and a small FCN.
    Note: variable names are reused (input_layer) — this keeps parity with the original code
    but can be confusing; comments clarify each step.
    """
    # ---------- Attention-ish block ----------
    # The model builds two input placeholders intended for query and value sequences.
    # Here we declare input tensors that would normally represent token ids for sequences.
    query_input = tf.keras.Input(shape=(input_layer,), dtype='float32')  # 'query' token ids placeholder
    value_input = tf.keras.Input(shape=(input_layer,), dtype='float32')  # 'value' token ids placeholder

    # Token embedding: maps token ids -> embedding vectors (embedding dim == input_layer)
    # Note: input_dim should be the vocabulary size; vocab is expected to be a dict so len(vocab) is used.
    token_embedding = tf.keras.layers.Embedding(input_dim=len(vocab), output_dim=input_layer)

    # Apply the same embedding layer to both query and value inputs
    query_embeddings = token_embedding(query_input)
    value_embeddings = token_embedding(value_input)

    # Small 1D CNN to produce sequence encodings (same layer applied to both query and value)
    cnn_layer = tf.keras.layers.Conv1D(filters=filter, kernel_size=4, padding='same')
    query_seq_encoding = cnn_layer(query_embeddings)
    value_seq_encoding = cnn_layer(value_embeddings)

    # Attention layer combining query and value sequence encodings (dot-product style)
    query_value_attention_seq = tf.keras.layers.Attention()([query_seq_encoding, value_seq_encoding])

    # Pool sequence encodings to fixed-size vectors
    query_encoding = tf.keras.layers.GlobalAveragePooling1D()(query_seq_encoding)
    query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(query_value_attention_seq)

    # Concatenate query encoding and the attention output to form a single vector input to FCN
    concatenated = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])

    # ---------- Fully Connected Network (FCN) ----------
    # The following reuses the name input_layer as a Keras Input layer from the concatenated tensor.
    # This pattern uses the functional API to wrap an existing tensor.
    input_layer = Input(shape=(input_layer,), tensor=concatenated)  # Keras Input that uses the concatenated tensor

    # Dense hidden layers with ReLU activations
    hidden_layer_one = Dense(hidden_layer_one, activation="relu")(input_layer)
    hidden_layer_two = Dense(hidden_layer_two, activation="relu")(hidden_layer_one)

    # Output layer with sigmoid for multi-label / multi-class probabilities
    output_layer = Dense(total_classes, activation="sigmoid")(hidden_layer_two)

    # Define model and compile with optimizer, loss and metrics (including F1 from tf-addons)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer='adam',
        loss='mse',  # mean squared error used here (common for regression; for classification consider 'binary_crossentropy')
        metrics=[
            "accuracy",
            "binary_accuracy",
            tfa.metrics.F1Score(num_classes=total_classes, average='weighted')
        ]
    )
    return model

# Different training set fractions to experiment with (100%, 75%, 50%, 25%)
training_sizes = [1.0, 0.75, 0.5, 0.25]
for size in training_sizes:
    logger.info(f"\nTraining with {size*100}% of the training data...")

    # Create a subset of the training data if size < 1.0, otherwise use all training data
    if size < 1.0:
        sub_train_x, _, sub_train_y, _ = train_test_split(train_x, train_y, train_size=size, random_state=42)
    else:
        sub_train_x, sub_train_y = train_x, train_y

    # K-Fold cross-validation over the chosen subset
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_no = 1
    for train_index, val_index in kf.split(sub_train_x):
        logger.info(f"Training fold {fold_no}...")
        # Prepare fold-specific training and validation splits
        fold_train_x, fold_val_x = sub_train_x[train_index], sub_train_x[val_index]
        fold_train_y, fold_val_y = sub_train_y.iloc[train_index], sub_train_y.iloc[val_index]

        logger.info(f"Fold {fold_no} - Training set size: {len(fold_train_x)}, Validation set size: {len(fold_val_x)}")

        # Early stopping callback to avoid overfitting if validation loss doesn't improve
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        # Create a new model instance for this fold
        model = create_model(input_layer, hidden_layer_one, hidden_layer_two, total_classes)

        # Fit model on this fold's training data and validate on the fold's validation set
        model.fit(
            fold_train_x, fold_train_y,
            epochs=40,
            batch_size=128,
            validation_data=(fold_val_x, fold_val_y),
            callbacks=[callback, ModelLogger()]  # Log epoch metrics and apply early stopping
        )

        # Print model summary (architecture and parameter counts)
        model.summary()

        # Save the trained model for this fold and training size
        model.save('dynamic_dataset_trained/fold_' + str(fold_no) + 'ds_' + str(size) + '.keras')

        # Evaluate the trained model on the held-out fixed test set
        evaluation_metrics = model.evaluate(test_x, test_y, batch_size=128)
        test_loss = evaluation_metrics[0]
        test_accuracy = evaluation_metrics[1]
        test_binary_accuracy = evaluation_metrics[2]
        test_f1_score = evaluation_metrics[3]

        # Log evaluation results
        logger.info(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}, Test binary accuracy: {test_binary_accuracy}, Test F1 score: {test_f1_score}")
        logger.info(f"[Info...] Fold {fold_no} completed!")
        fold_no += 1

# ------------------------------------------------------------------
# The following commented-out code blocks show alternative training/prediction flows:
# - A KFold training loop over the entire dataset (commented)
# - A helper prediction function that encodes a single text and predicts classes (commented)
# Keep them for reference or future use.
# ------------------------------------------------------------------

# for train_index, test_index in kf.split(sentence_embeddings):
#     logger.info(f"[Info...] Training fold {fold_no}...")
#     train_x, test_x = sentence_embeddings[train_index], sentence_embeddings[test_index]
#     train_y, test_y = multi_label_df.iloc[train_index], multi_label_df.iloc[test_index]
#     
#     model = create_model(input_layer, hidden_layer_one, hidden_layer_two, total_classes)
#
#     # Train model
#     callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
#     trained_model = model.fit(train_x, train_y, epochs=40, batch_size=128, validation_data=(test_x, test_y), callbacks=[callback, ModelLogger()])
#
#     model.summary()
#
#     model.save('dynamic_dataset_trained/fold_' + str(fold_no) + '.keras')
#     logger.info(f"[Info...] Fold {fold_no} completed!")
#
#     fold_no += 1
#
# def prediction(text):
#     # Example helper to run a prediction on a single text string
#     text_list = []
#     text_list.append(text)
#     encoded_text = sbert.encode(text_list)
#     prediction_list = model.predict(encoded_text)[0]
#
#     class_names = multi_label.classes_
#     predicted_class_dict = {}
#     print(class_names)
#
#     for i in range(0, len(class_names)):
#         predicted_class_dict[class_names[i]] = prediction_list[i]
#     
#     for key in predicted_class_dict:
#         print(f"Class - {key}: {round(predicted_class_dict[key]*100, 2)}")
#         
# sentences = [
#     "I feel secure knowing I can rely on you.",
#     "I had a blast playing board games with you last night.",
#     "That's fine. ",
#     "You have the freedom to be yourself.",
#     "I should dump you out on the street.",
#     "You're not good enough for me.",
#     "I've disowned my whole family. They never understood me.",
#     "Why didn't you answer my call immediately? Who were you with?",
#     "If you really loved me, you'd do this.",
#     "you're lucky I give you my time.",
#     "You shouldn't say no to me. Don't you trust me.",
#     "Flirting doesn't mean anything.",
#     "I noticed we're running low on detergent, I'll pick some up on my way back.",
#     "I've disowned my whole family. They never understood me.",
#     "I am sorry",
#     "You have the freedom to be yourself.",
#     # ... additional example sentences
# ]
#
# for sentence in sentences:
#     print(f"[ ================ {sentence} ============== ]")
#     prediction(sentence)
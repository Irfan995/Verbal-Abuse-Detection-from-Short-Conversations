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

# Configure logging
logging.basicConfig(filename='dynamic_dataset_logs/violence_detection_k_folded.log', level=logging.INFO)
logger = logging.getLogger()

class ModelLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logger.info(f'Epoch {epoch+1}: {logs}')

def calculate_layer_input(dimension):
    filter = (dimension * 2) / 4
    input_layer = dimension
    hidden_layer_one = (dimension * 2) / 3
    hidden_layer_two = dimension / 3
    return int(filter), int(input_layer), int(hidden_layer_one), int(hidden_layer_two)
filter, input_layer, hidden_layer_one, hidden_layer_two = calculate_layer_input(768)

# C:\Users\fai94s\Documents\Context Detection Codes\datasets\domestic_violence_dataset.xlsx
df = pd.read_excel(r"domestic_violence_dataset.xlsx")
df = df[['Text', 'Flag_type']]

df['Text'] = df['Text'].str.lower()

# Label Binarization
multi_label = LabelBinarizer()
labels = multi_label.fit_transform(df['Flag_type'])
total_classes = len(multi_label.classes_)

# Coverting to floating value
multi_label_df = pd.DataFrame(labels, columns=multi_label.classes_)
np.asarray(multi_label_df).astype('float32').reshape((-1,1))

# Encoding voice sample to feed in the network
voice_list = df['Text'].to_numpy()
sbert = SentenceTransformer('all-distilroberta-v1')
sentence_embeddings = sbert.encode(voice_list)

tokenizer = sbert.tokenizer
vocab = tokenizer.vocab

# Split dataset into train and test
train_x, test_x, train_y, test_y = train_test_split(sentence_embeddings, 
                                                      multi_label_df, 
                                                      train_size=0.7, 
                                                      test_size=0.3, 
                                                      random_state=42,
                                                      stratify=df['Flag_type'])

def create_model(input_layer, hidden_layer_one, hidden_layer_two, total_classes):
    # Attention Layer - starts
    query_input = tf.keras.Input(shape=(input_layer,), dtype='float32')  # dq
    value_input = tf.keras.Input(shape=(input_layer,), dtype='float32')  # dv
    token_embedding = tf.keras.layers.Embedding(input_dim=len(vocab), output_dim=input_layer)  # dk
    query_embeddings = token_embedding(query_input)  # Embedding vector
    value_embeddings = token_embedding(value_input)  # Embedding vector

    # CNN layer as input
    cnn_layer = tf.keras.layers.Conv1D(
        filters=filter,
        kernel_size=4,
        padding='same')  
    query_seq_encoding = cnn_layer(query_embeddings)
    value_seq_encoding = cnn_layer(value_embeddings)
    query_value_attention_seq = tf.keras.layers.Attention()([query_seq_encoding, value_seq_encoding])  # First MatMul
    query_encoding = tf.keras.layers.GlobalAveragePooling1D()(query_seq_encoding)
    query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(query_value_attention_seq)
    input_layer = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])
    # Attention Layer - ends

    # Creating layers for FCN
    input_layer = Input(shape=(input_layer,), tensor=input_layer)
    hidden_layer_one = Dense(hidden_layer_one, activation="relu")(input_layer)
    hidden_layer_two = Dense(hidden_layer_two, activation="relu")(hidden_layer_one)
    output_layer= Dense(total_classes, activation="sigmoid")(hidden_layer_two)

    # Defining the model by specifying the input and output layers
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer='adam',
        loss='mse', 
        metrics=["accuracy",
                "binary_accuracy",
                tfa.metrics.F1Score(num_classes=total_classes, average='weighted')
                ]
    )
    return model

training_sizes = [1.0, 0.75, 0.5, 0.25]
for size in training_sizes:
    logger.info(f"\nTraining with {size*100}% of the training data...")

    # Create subset of the training data
    if size < 1.0:
        sub_train_x, _, sub_train_y, _ = train_test_split(train_x, train_y, train_size=size, random_state=42)
    else:
        sub_train_x, sub_train_y = train_x, train_y

    # K-Fold Cross Validation
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_no = 1
    for train_index, val_index in kf.split(sub_train_x):  # train_x is your entire training dataset
        logger.info(f"Training fold {fold_no}...")
        fold_train_x, fold_val_x = sub_train_x[train_index], sub_train_x[val_index]
        fold_train_y, fold_val_y = sub_train_y.iloc[train_index], sub_train_y.iloc[val_index]
        
        logger.info(f"Fold {fold_no} - Training set size: {len(fold_train_x)}, Validation set size: {len(fold_val_x)}")
        
        # Create and compile your model here
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        model = create_model(input_layer, hidden_layer_one, hidden_layer_two, total_classes)
        
        # Train the model on the current fold's training data
        model.fit(
            fold_train_x, fold_train_y,
            epochs=40,
            batch_size=128,
            validation_data=(fold_val_x, fold_val_y),
            callbacks=[callback, ModelLogger()]  # Optional: use EarlyStopping to prevent overfitting
        )
        model.summary()
        model.save('dynamic_dataset_trained/fold_' + str(fold_no) + 'ds_' + str(size) + '.keras')
        # Evaluate the model on the fixed test set
        evaluation_metrics = model.evaluate(test_x, test_y, batch_size=128)
        test_loss = evaluation_metrics[0]
        test_accuracy = evaluation_metrics[1]
        test_binary_accuracy = evaluation_metrics[2]
        test_f1_score = evaluation_metrics[3]

        logger.info(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}, Test binary accuracy: {test_binary_accuracy}, Test F1 score: {test_f1_score}")
        logger.info(f"[Info...] Fold {fold_no} completed!")
        fold_no += 1

# for train_index, test_index in kf.split(sentence_embeddings):
#     logger.info(f"[Info...] Training fold {fold_no}...")
#     train_x, test_x = sentence_embeddings[train_index], sentence_embeddings[test_index]
#     train_y, test_y = multi_label_df.iloc[train_index], multi_label_df.iloc[test_index]
    
#     model = create_model(input_layer, hidden_layer_one, hidden_layer_two, total_classes)

#     # Train model
#     callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
#     trained_model = model.fit(train_x, train_y, epochs=40, batch_size=128, validation_data=(test_x, test_y), callbacks=[callback, ModelLogger()])

#     model.summary()

#     model.save('dynamic_dataset_trained/fold_' + str(fold_no) + '.keras')
#     logger.info(f"[Info...] Fold {fold_no} completed!")

#     fold_no += 1

# def prediction(text):
#     text_list = []
#     text_list.append(text)
#     encoded_text = sbert.encode(text_list)
#     prediction_list = model.predict(encoded_text)[0]

#     class_names = multi_label.classes_
#     predicted_class_dict = {}
#     print(class_names)

#     for i in range(0, len(class_names)):
#         predicted_class_dict[class_names[i]] = prediction_list[i]
    
#     for key in predicted_class_dict:
#         print(f"Class - {key}: {round(predicted_class_dict[key]*100, 2)}")
        
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
    # "You are going to regret saying no",
    # "stop talking to other guys",
    # "Let's play Russian Roulette",
    # "If you really loved me, you would not go out tonight",
    # "You are amazing. Thank you",
    # "I know I hurt your feelings. I'm sorry",
    # "Take out the trash, please",
    # "Could you pass me the salt?",
    # "you are a stupid bitch",
    # "I'll cut you",
    # "I love you so much. If you leave, I will hurt myself",
    # "I want you, you can't say no.",
    # "There's one in the chamber.  You're turn with the revolver.",
    # "I'll shoot you",
    # "Shoot yourself",
    # "you are trash",
    # "Could you pass me the pepper?",
    # "Take out the trash, please."
# ]

# for sentence in sentences:
#     print(f"[ ================ {sentence} ============== ]")
#     prediction(sentence)
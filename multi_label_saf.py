import logging
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalAveragePooling1D, Attention, Embedding, Concatenate
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
from contextlib import redirect_stdout

# Configure logging
logging.basicConfig(filename='w_violence_detection_b32_soft.log', level=logging.INFO)
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

# Load data
df = pd.read_excel(r"datasets\family_conflict_conv.xlsx")
df = df[['Voice', 'Context', 'Sentiment']]  # Assuming 'Sentiment' is the new categorical column

df['Voice'] = df['Voice'].str.lower()

# Label Binarization for context
multi_label = LabelBinarizer()
labels = multi_label.fit_transform(df['Context'])
total_classes = len(multi_label.classes_)

# Convert context labels to floating value
multi_label_df = pd.DataFrame(labels, columns=multi_label.classes_)
multi_label_df = np.asarray(multi_label_df).astype('float32')

# Encode voice sample to feed in the network
voice_list = df['Voice'].to_numpy()
sbert = SentenceTransformer('all-distilroberta-v1')
sentence_embeddings = sbert.encode(voice_list)

# One-hot encoding the categorical 'Sentiment' column
label_encoder = LabelBinarizer()
integer_encoded_sentiments = label_encoder.fit_transform(df['Sentiment'])
sentiment_df = pd.DataFrame(integer_encoded_sentiments, columns=label_encoder.classes_)
# sentiment_df = np.asarray(sentiment_df).astype('float32')
# sentiment_df = sbert.encode(sentiment_df)
sentiment_list = df['Sentiment'].to_numpy()
sentiment_embedding = sbert.encode(sentiment_list)

tokenizer = sbert.tokenizer
vocab = tokenizer.vocab

# Split dataset into train and test
train_x_text, test_x_text, train_x_sentiment, test_x_sentiment, train_y, test_y = train_test_split(
    sentence_embeddings, 
    sentiment_embedding, 
    multi_label_df, 
    train_size=0.7, 
    test_size=0.3, 
    random_state=42
)

print(train_x_text.shape, test_x_text.shape, train_x_sentiment.shape, test_x_sentiment.shape, train_y.shape, test_y.shape)

# Attention Layer - starts
query_input = Input(shape=(input_layer,), dtype='float32')  # dq
value_input = Input(shape=(input_layer,), dtype='float32')  # dv
token_embedding = Embedding(input_dim=len(vocab), output_dim=input_layer)  # dk
query_embeddings = token_embedding(query_input)  # Embedding vector
value_embeddings = token_embedding(value_input)  # Embedding vector

# CNN layer as input
cnn_layer = Conv1D(
    filters=filter,
    kernel_size=4,
    padding='same'
)  
query_seq_encoding = cnn_layer(query_embeddings)
value_seq_encoding = cnn_layer(value_embeddings)
query_value_attention_seq = Attention()([query_seq_encoding, value_seq_encoding])  # First MatMul
query_encoding = GlobalAveragePooling1D()(query_seq_encoding)
query_value_attention = GlobalAveragePooling1D()(query_value_attention_seq)
attention_input_layer = Concatenate()([query_encoding, query_value_attention])

# Assign weights to contexts
class_frequencies = np.sum(train_y, axis=0)
total_samples = len(train_y)
class_weights = total_samples / (len(class_frequencies) * class_frequencies)

class_weights_dict = {idx: weight for idx, weight in enumerate(class_weights)}

# Creating layers for FCN

text_input_layer = Input(shape=(input_layer,), tensor=attention_input_layer, name='text_input')

# Categorical input layer
input_sentiment = Input(shape=(input_layer,), name='sentiment_input')

# Concatenate text and categorical inputs
concatenated_inputs = Concatenate()([text_input_layer, input_sentiment])
# input_layer = Input(shape=(384,), tensor=concatenated_inputs)
hidden_layer_one = Dense(hidden_layer_one, activation="relu")(concatenated_inputs)
hidden_layer_two = Dense(hidden_layer_two, activation="relu")(hidden_layer_one)
output_layer= Dense(total_classes, activation="softmax")(hidden_layer_two)

# Defining the model by specifying the input and output layers
model = Model(inputs=[text_input_layer, input_sentiment], outputs=output_layer)
model.compile(
    optimizer='adam',
    loss='mse', metrics=["accuracy","binary_accuracy",tfa.metrics.F1Score(num_classes=total_classes,average='weighted')]
)

# Train model
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
trained_model = model.fit([train_x_text, train_x_sentiment], train_y, class_weight=class_weights_dict, epochs=40, batch_size=32, validation_data=([test_x_text, test_x_sentiment], test_y), callbacks=[callback, ModelLogger()])

model.summary()

model.save('w_violence_detection_b32_soft_model.keras')


def prediction(text, sentiment):
    text_list = [text]
    sentiment_list = [sentiment]
    encoded_text = sbert.encode(text_list)
    encoded_sentiment = sbert.encode(sentiment_list)
    
    sentiment_array = np.array([sentiment])
    sentiment_encoded = label_encoder.transform(sentiment_array)
    # sentiment_one_hot = tf.keras.utils.to_categorical(sentiment_encoded, num_classes=sentiment_categories.shape[1])
    
    prediction_list = model.predict([encoded_text, encoded_sentiment])[0]

    class_names = multi_label.classes_
    predicted_class_dict = {class_names[i]: prediction_list[i] for i in range(len(class_names))}
    
    for key, value in predicted_class_dict.items():
        logger.info(f"Class - {key}: {round(value * 100, 2)}")

# while True:
#     print("\n------------Testing---------------\n")
#     text = input("Enter statement to detect context or enter 'E' to exit:").lower()
#     if text == "e":
#         break
#     sentiment = input("Enter the sentiment (e.g., happy, sad, angry):").lower()
#     prediction(text, sentiment)
    
sentences = [
    {"voice": "You are going to regret saying no", "sentiment": "angry"},
    {"voice": "stop talking to other guys", "sentiment": "angry"},
    {"voice": "Let's play Russian Roulette","sentiment": "angry"},
    {"voice": "Let's play Russian Roulette","sentiment": "normal"},
    {"voice": "If you really loved me, you would not go out tonight", "sentiment": "angry"},
    {"voice": "You are amazing. Thank you", "sentiment": "happy"},
    {"voice": "You are amazing. Thank you", "sentiment": "angry"},
    {"voice": "I know I hurt your feelings. I'm sorry", "sentiment": "normal"},
    {"voice": "Take out the trash, please", "sentiment": "normal"},
    {"voice": "Could you pass me the salt?", "sentiment": "normal"},
    {"voice": "you are a stupid bitch", "sentiment": "normal"},
    {"voice": "I'll cut you", "sentiment": "normal"},
    {"voice": "I love you so much. If you leave, I will hurt myself", "sentiment": "normal"},
    {"voice": "I want you, you can't say no.", "sentiment": "normal"},
    {"voice": "I'll shoot you", "sentiment": "normal"},
    {"voice": "Shoot yourself", "sentiment": "normal"},
    {"voice": "you are trash", "sentiment": "happy"},
    {"voice": "you are trash", "sentiment": "angry"},
    {"voice": "you are trash", "sentiment": "normal"},
    {"voice": "Could you pass me the pepper?", "sentiment": "normal"},
    {"voice": "Take out the trash, please.", "sentiment": "normal"},
    {"voice": "Stop talking to me", "sentiment": "normal"}
]

for sentence in sentences:
    logger.info(f"[ ================ {sentence} ============== ]")
    prediction(sentence["voice"], sentence["sentiment"])
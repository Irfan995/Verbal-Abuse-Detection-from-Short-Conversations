import logging
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalAveragePooling1D, Attention, Embedding, Concatenate
from tensorflow.keras.models import Model
import tensorflow_addons as tfa

# Configure logging to a file for training/inference outputs
logging.basicConfig(filename='w_violence_detection_b32_soft.log', level=logging.INFO)
logger = logging.getLogger()

# Keras callback that logs epoch metrics to the configured logger
class ModelLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logger.info(f'Epoch {epoch+1}: {logs}')

# Helper to compute layer sizes from embedding dimension (768 for SBERT)
def calculate_layer_input(dimension):
    # compute filter and hidden layer sizes based on heuristics
    filter = (dimension * 2) / 4
    input_layer = dimension
    hidden_layer_one = (dimension * 2) / 3
    hidden_layer_two = dimension / 3
    return int(filter), int(input_layer), int(hidden_layer_one), int(hidden_layer_two)

def load_and_preprocess(dataset_path, sbert_model_name='all-distilroberta-v1'):
    """
    Load dataset, normalize text, compute SBERT embeddings for voice and sentiment,
    and prepare multi-label and sentiment encodings.
    Returns: sentence_embeddings, sentiment_embeddings, multi_label_df, label_encoder, multi_label, df, sbert
    """
    df = pd.read_excel(dataset_path)
    df = df[['Voice', 'Context', 'Sentiment']]  # LABEL_COLUMNS
    df['Voice'] = df['Voice'].str.lower()

    # Multi-label (context) binarization
    multi_label = LabelBinarizer()
    labels = multi_label.fit_transform(df['Context'])
    total_classes = len(multi_label.classes_)
    multi_label_df = np.asarray(pd.DataFrame(labels, columns=multi_label.classes_)).astype('float32')

    # SBERT embeddings for voice and sentiment
    sbert = SentenceTransformer(sbert_model_name)
    voice_list = df['Voice'].to_numpy()
    sentence_embeddings = sbert.encode(voice_list)  # (N, 768)

    sentiment_list = df['Sentiment'].to_numpy()
    sentiment_embedding = sbert.encode(sentiment_list)  # (N, 768)

    # LabelBinarizer for sentiment (one-hot/binary)
    sentiment_label_binarizer = LabelBinarizer()
    sentiment_label_binarizer.fit_transform(df['Sentiment'])

    return sentence_embeddings, sentiment_embedding, multi_label_df, sentiment_label_binarizer, multi_label, df, sbert

def build_model(input_layer_dim, vocab_size, total_classes, filter_size, hidden_one, hidden_two):
    """
    Build the Keras model using the same architecture:
    - token embedding -> Conv1D -> Attention -> pooling
    - concatenate pooled text encoding with sentiment embedding
    - dense hidden layers -> output
    Note: this preserves original wiring (token embedding applied to inputs).
    """
    # Inputs representing query/value for attention (shape matches SBERT embedding dim)
    query_input = Input(shape=(input_layer_dim,), dtype='float32')  # dq
    value_input = Input(shape=(input_layer_dim,), dtype='float32')  # dv

    # Token embedding layer (maps integer token ids to vectors). Kept for architecture completeness.
    token_embedding = Embedding(input_dim=vocab_size, output_dim=input_layer_dim)  # dk

    # Apply embedding layer to the inputs (treats inputs as token indices, though actual inputs are embeddings)
    query_embeddings = token_embedding(query_input)
    value_embeddings = token_embedding(value_input)

    # Simple 1D CNN applied on embedded sequences
    cnn_layer = Conv1D(filters=filter_size, kernel_size=4, padding='same')
    query_seq_encoding = cnn_layer(query_embeddings)
    value_seq_encoding = cnn_layer(value_embeddings)

    # Attention between query and value CNN outputs
    query_value_attention_seq = Attention()([query_seq_encoding, value_seq_encoding])

    # Pool sequence encodings to fixed-size vectors
    query_encoding = GlobalAveragePooling1D()(query_seq_encoding)
    query_value_attention = GlobalAveragePooling1D()(query_value_attention_seq)

    # Concatenate pooled query encoding and pooled attention output
    attention_input_layer = Concatenate()([query_encoding, query_value_attention])

    # Define an Input layer for text that uses the precomputed attention representation via `tensor=...`
    text_input_layer = Input(shape=(input_layer_dim,), tensor=attention_input_layer, name='text_input')

    # Sentiment input: we feed SBERT sentiment embeddings directly as this input
    input_sentiment = Input(shape=(input_layer_dim,), name='sentiment_input')

    # Concatenate text encoding and sentiment embedding
    concatenated_inputs = Concatenate()([text_input_layer, input_sentiment])

    # Two hidden dense layers followed by final softmax for multi-class (or multi-label) output
    hidden_layer_one = Dense(hidden_one, activation="relu")(concatenated_inputs)
    hidden_layer_two = Dense(hidden_two, activation="relu")(hidden_layer_one)
    output_layer = Dense(total_classes, activation="softmax")(hidden_layer_two)

    model = Model(inputs=[text_input_layer, input_sentiment], outputs=output_layer)
    model.compile(
        optimizer='adam',
        loss='mse',  # mean squared error used here; consider 'binary_crossentropy' for multi-label tasks
        metrics=["accuracy", "binary_accuracy", tfa.metrics.F1Score(num_classes=total_classes, average='weighted')]
    )
    return model

def train_model(model, train_x_text, train_x_sentiment, train_y, test_x_text, test_x_sentiment, test_y, epochs=40, batch_size=32):
    """
    Train the model with class weighting to handle imbalance and return the trained model and history.
    """
    # Compute class frequencies and weights from training labels
    class_frequencies = np.sum(train_y, axis=0)
    total_samples = len(train_y)
    class_weights = total_samples / (len(class_frequencies) * class_frequencies)
    class_weights_dict = {idx: weight for idx, weight in enumerate(class_weights)}

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    trained = model.fit(
        [train_x_text, train_x_sentiment],
        train_y,
        class_weight=class_weights_dict,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([test_x_text, test_x_sentiment], test_y),
        callbacks=[callback, ModelLogger()]
    )
    return trained

def prediction(model, sbert, label_encoder, multi_label, text, sentiment):
    """
    Encode text and sentiment with SBERT and run the model to produce class scores.
    Also logs class probabilities to the logger.
    """
    text_list = [text]
    sentiment_list = [sentiment]

    encoded_text = sbert.encode(text_list)
    encoded_sentiment = sbert.encode(sentiment_list)

    # Run model prediction; model expects two inputs: text embedding and sentiment embedding
    prediction_list = model.predict([encoded_text, encoded_sentiment])[0]

    class_names = multi_label.classes_
    predicted_class_dict = {class_names[i]: prediction_list[i] for i in range(len(class_names))}

    for key, value in predicted_class_dict.items():
        logger.info(f"Class - {key}: {round(value * 100, 2)}")

# -------------------------
# Execution / main flow
# -------------------------
if __name__ == '__main__':
    # Load & preprocess data
    dataset_path = r"datasets\family_conflict_conv.xlsx"
    sentence_embeddings, sentiment_embedding, multi_label_df, sentiment_label_binarizer, multi_label, df, sbert = load_and_preprocess(dataset_path)

    # Model sizing based on SBERT embedding dim (768)
    filter_size, input_layer, hidden_layer_one, hidden_layer_two = calculate_layer_input(768)
    total_classes = len(multi_label.classes_)

    # Tokenizer vocab (used for Embedding layer size)
    tokenizer = sbert.tokenizer
    vocab = tokenizer.vocab

    # Train/test split
    train_x_text, test_x_text, train_x_sentiment, test_x_sentiment, train_y, test_y = train_test_split(
        sentence_embeddings,
        sentiment_embedding,
        multi_label_df,
        train_size=0.7,
        test_size=0.3,
        random_state=42
    )

    print(train_x_text.shape, test_x_text.shape, train_x_sentiment.shape, test_x_sentiment.shape, train_y.shape, test_y.shape)

    # Build model
    model = build_model(
        input_layer_dim=input_layer,
        vocab_size=len(vocab),
        total_classes=total_classes,
        filter_size=filter_size,
        hidden_one=hidden_layer_one,
        hidden_two=hidden_layer_two
    )

    # Train
    history = train_model(model, train_x_text, train_x_sentiment, train_y, test_x_text, test_x_sentiment, test_y)

    model.summary()
    model.save('w_violence_detection_b32_soft_model.keras')

    # -------------------------
    # Example usage / quick tests
    # -------------------------
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
        prediction(model, sbert, sentiment_label_binarizer, multi_label, sentence["voice"], sentence["sentiment"])

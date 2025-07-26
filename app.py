import streamlit as st
import tensorflow as tf
import numpy as np
import re
from transformers import DistilBertTokenizer, TFDistilBertModel
from tensorflow.keras.layers import Conv1D, DepthwiseConv1D, LayerNormalization, Dropout, Add, Average, GlobalAveragePooling1D, Dense, Multiply
from tensorflow.keras.models import load_model as keras_load_model

# ----- Custom Layers -----
def MultiScaleConvBlock_fn(filters):
    def layer(x):
        x1 = Conv1D(filters, 3, padding='same', activation='relu')(x)
        x2 = Conv1D(filters, 5, padding='same', activation='relu')(x)
        x3 = Conv1D(filters, 7, padding='same', activation='relu')(x)
        return Average()([x1, x2, x3])
    return layer

class ConvMixerBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, dropout_rate=0.1, dilation_rate=2, **kwargs):
        super(ConvMixerBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.dilation_rate = dilation_rate

        self.depthwise = DepthwiseConv1D(kernel_size, padding='same', dilation_rate=dilation_rate)
        self.pointwise = Conv1D(filters, 1, activation='relu')
        self.layernorm = LayerNormalization()
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs):
        x = self.depthwise(inputs)
        x = self.layernorm(x)
        x = self.pointwise(x)
        x = self.dropout(x)
        return Add()([x, inputs])

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate,
            'dilation_rate': self.dilation_rate
        })
        return config

class ChannelSelfAttention(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ChannelSelfAttention, self).__init__(**kwargs)
        self.filters = filters
        self.global_avg_pool = GlobalAveragePooling1D()
        self.dense1 = Dense(filters // 8, activation='relu')
        self.dense2 = Dense(filters, activation='sigmoid')

    def call(self, inputs):
        x = self.global_avg_pool(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return Multiply()([inputs, tf.expand_dims(x, axis=1)])

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})
        return config

# ----- Load Model & Tokenizer -----
custom_objects = {
    'ConvMixerBlock': ConvMixerBlock,
    'ChannelSelfAttention': ChannelSelfAttention,
}

@st.cache_resource
def load_model():
    return keras_load_model('Hybrid_Convmix.h5', custom_objects=custom_objects)

@st.cache_resource
def load_tokenizer_and_bert():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    return tokenizer, bert_model

# ----- Text Cleaning -----
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ----- Get CLS Token Embedding -----
def get_embedding(text, tokenizer, bert_model):
    inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True, max_length=128)
    outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
    return cls_embedding.numpy().squeeze()  # shape: (768,)

# ----- Streamlit UI -----
st.title("Suicide Prediction from Text Using DistilBERT + Custom Model")

tokenizer, bert_model = load_tokenizer_and_bert()
model = load_model()

user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if not user_input.strip():
        st.error("Please enter some text for prediction.")
    else:
        cleaned_text = clean_text(user_input)
        st.write(f"Cleaned Text: {cleaned_text}")

        # Get CLS token embedding and reshape
        embedding_vector = get_embedding(cleaned_text, tokenizer, bert_model)
        embedding_vector = np.expand_dims(embedding_vector, axis=0)     # [1, 768]
        embedding_vector = np.expand_dims(embedding_vector, axis=-1)    # [1, 768, 1]

        # Predict
        prediction_prob = model.predict(embedding_vector)[0][0]
        prediction_label = "Suicide" if prediction_prob > 0.1 else "Non-Suicide"

        # Show result
        st.write(f"Prediction Probability: {prediction_prob:.4f}")
        st.success(f"Predicted Class: **{prediction_label}**")


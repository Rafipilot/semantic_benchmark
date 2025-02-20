import tensorflow_datasets as tfds
from openai import OpenAI
import ao_embeddings.binaryEmbeddings as be
import ao_core as ao
import ao_arch as ar
from dotenv import load_dotenv
import os
import streamlit as st
import numpy as np

# Load environment variables
load_dotenv()
api_key = os.environ.get("openai_api_key")

# Initialize agent in session_state if not already present


arch_i, arch_z, arch_c = [128], [10], [0]
connector_function = "full_conn"
Arch = ar.Arch(arch_i, arch_z, arch_c, connector_function, description="none")
agent = ao.Agent(Arch)
agent.full_conn_compress = True


if "agent" not in st.session_state:

    st.session_state.agent = agent

# Initialize the binary embeddings instance in session_state if not already present
if "be" not in st.session_state:
    st.session_state.be = be.binaryEmbeddings(api_key, cacheName="cache.json", numberBinaryDigits=128)

# Load the IMDb dataset into session_state if not already present
if "train_data" not in st.session_state or "test_data" not in st.session_state:
    dataset, info = tfds.load("imdb_reviews", split=["train", "test"], as_supervised=True, with_info=True)
    st.session_state.train_data, st.session_state.test_data = dataset[0], dataset[1]

batch_size = 1000
if "batch_numbers" not in st.session_state:
    st.session_state.batch_numbers = 1

def train_agent():
    texts, labels = [], []
    # Take a batch from the training data
    for i in range(st.session_state.batch_numbers):
        num = i*batch_size
        for text, label in st.session_state.train_data.skip(num).take(batch_size):
            texts.append(text.numpy().decode('utf-8'))
            labels.append([int(label.numpy())])
            
        # Get embeddings and convert them to binary
        embeddings = st.session_state.be.get_embedding_batch(texts)
        binary_embeddings = [st.session_state.be.embeddingToBinary(embedding) for embedding in embeddings]
    
    # Train the agent on the batch
    st.session_state.agent.next_state_batch(
        INPUT=binary_embeddings,
        LABEL=labels,
        unsequenced=True,
        print_result=True
    )
    st.success("Training complete. Start testing...")

def test_agent():
    test_texts, test_labels = [], []
    
    # Take a sample from the test data
    for text, label in st.session_state.test_data.take(100):
        test_texts.append(text.numpy().decode('utf-8'))
        test_labels.append(label.numpy())
        
    # Get embeddings and convert them to binary
    test_embeddings = st.session_state.be.get_embedding_batch(test_texts)
    test_embeddings_binary = [st.session_state.be.embeddingToBinary(emb) for emb in test_embeddings]

    success = 0

    # Evaluate the agent on each test sample
    for i, binary_embedding in enumerate(test_embeddings_binary):
        response = st.session_state.agent.next_state(binary_embedding, unsequenced=True)
        st.session_state.agent.reset_state()
        prediction = 1 if sum(response) >= 5 else 0
        print("Test: ", test_texts[i])
        print("Predicted: ", prediction, "Actual: ", test_labels[i])
        result = (f"Test: {test_texts[i]}\nPredicted: {prediction}, Actual: {test_labels[i]}\n")
        print(result)
        if test_labels[i] == prediction:
            success += 1
            
    # Display results

    
    total = len(test_embeddings_binary)
    st.write(f"Success rate: {success}/{total} ({(success/total)*100:.2f}%)")


st.set_page_config(
    page_title="Sentiment Analysis with WNNs through Binary Embeddings",
    page_icon=":guardsman:",
    layout="wide",
    )


st.title("Sentiment Analysis with WNNs through Binary Embeddings")

# Button to train the agent
if st.button("Train Agent"):
    train_agent()
st.text("   ")
# Button to test the agent
if st.button("Test Agent with Dataset"):
    test_agent()

# Allow custom text input for sentiment analysis
col1, col2 = st.columns(2)

with col1:

    st.subheader("Test with Custom Text")
    user_input = st.text_area("Enter text for sentiment analysis:")

    if st.button("Analyze Sentiment"):
        if user_input:
            # Get embedding for the input text and convert to binary
            input_embedding = st.session_state.be.get_embedding(user_input)
            binary_embedding = st.session_state.be.embeddingToBinary(input_embedding)
            
            # Get prediction from the agent
            response = st.session_state.agent.next_state(binary_embedding, unsequenced=True)
            st.session_state.agent.reset_state()
            prediction = 1 if sum(response) >= 5 else 0
            sentiment = "Positive" if prediction == 1 else "Negative"
            st.success(f"Sentiment: {sentiment}")
        else:
            st.warning("Please enter some text to analyze.")


with col2:
    st.subheader("Train with custom text")
    custom_text = st.text_area("Enter custom text for training:")
    custom_label = st.selectbox("Select label (0 for Negative, 1 for Positive):", [0, 1])
    if custom_label == 0:
        label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    elif custom_label == 1:
        label = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  
    if st.button("Train with Custom Text"):
        if custom_text:
            # Get embedding for the input text and convert to binary
            input_embedding = st.session_state.be.get_embedding(custom_text)
            binary_embedding = st.session_state.be.embeddingToBinary(input_embedding)
            
            # Train the agent with the custom text and label
            st.session_state.agent.next_state(INPUT=binary_embedding,LABEL=label,unsequenced=True, print_result=True
            )
            st.success("Custom text trained successfully.")
        else:
            st.warning("Please enter some text to train.")
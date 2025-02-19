import tensorflow_datasets as tfds
from openai import OpenAI
import ao_embeddings.binaryEmbeddings as be
import ao_core as ao
import ao_arch as ar
from dotenv import load_dotenv
import os
import streamlit as st

# Load environment variables
load_dotenv()
api_key = os.environ.get("openai_api_key")

# Initialize agent in session_state if not already present


arch_i, arch_z, arch_c = [1536], [10], [0]
connector_function = "full_conn"
Arch = ar.Arch(arch_i, arch_z, arch_c, connector_function, description="none")
agent = ao.Agent(Arch)
agent.full_conn_compress = True


if "agent" not in st.session_state:

    st.session_state.agent = agent

# Initialize the binary embeddings instance in session_state if not already present
if "be" not in st.session_state:
    st.session_state.be = be.binaryEmbeddings(api_key, cacheName="cache.json", numberBinaryDigits=1536)

# Load the IMDb dataset into session_state if not already present
if "train_data" not in st.session_state or "test_data" not in st.session_state:
    dataset, info = tfds.load("imdb_reviews", split=["train", "test"], as_supervised=True, with_info=True)
    st.session_state.train_data, st.session_state.test_data = dataset[0], dataset[1]

batch_size = 2000

def train_agent():
    texts, labels = [], []
    # Take a batch from the training data
    for text, label in st.session_state.train_data.take(batch_size):
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
        DD=False,
        print_result=True
    )
    st.success("Training complete. Starting testing...")

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
        response = st.session_state.agent.next_state(binary_embedding, DD=False, unsequenced=True)
        st.session_state.agent.reset_state()
        prediction = 1 if sum(response) >= 5 else 0
        print("Test: ", test_texts[i])
        print("Predicted: ", prediction, "Actual: ", test_labels[i])
        result = (f"Test: {test_texts[i]}\nPredicted: {prediction}, Actual: {test_labels[i]}\n")
        if test_labels[i] == prediction:
            success += 1
            
    # Display results

    st.write(result)
    total = len(test_embeddings_binary)
    st.write(f"Success rate: {success}/{total} ({(success/total)*100:.2f}%)")

st.title("Sentiment Analysis with Session State")

# Button to train the agent
if st.button("Train Agent"):
    train_agent()

# Button to test the agent
if st.button("Test Agent with Dataset"):
    test_agent()

# Allow custom text input for sentiment analysis
st.subheader("Test with Custom Text")
user_input = st.text_area("Enter text for sentiment analysis:")

if st.button("Analyze Sentiment"):
    if user_input:
        # Get embedding for the input text and convert to binary
        input_embedding = st.session_state.be.get_embedding(user_input)
        binary_embedding = st.session_state.be.embeddingToBinary(input_embedding)
        
        # Get prediction from the agent
        response = st.session_state.agent.next_state(binary_embedding, DD=False, unsequenced=True)
        st.session_state.agent.reset_state()
        prediction = 1 if sum(response) >= 5 else 0
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.success(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text to analyze.")

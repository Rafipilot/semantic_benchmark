import tensorflow_datasets as tfds
from openai import OpenAI
import ao_embeddings.binaryEmbeddings as be
import ao_core as ao
import ao_arch as ar
from dotenv import load_dotenv
import os
import numpy as np
import re


np.random.seed(42)


def clean_text(text):
    return text

load_dotenv()
api_key = os.environ.get("openai_api_key")

arch_i, arch_z, arch_c = [128], [20], [0]
connector_function = "full_conn"
Arch = ar.Arch(arch_i, arch_z, arch_c, connector_function, description="none")
agent = ao.Agent(Arch)
agent.full_conn_compress = True

client = OpenAI(api_key=api_key)
be = be.binaryEmbeddings(api_key, cacheName="cache.json", numberBinaryDigits=128)

batch_size = 1000

batch_numbers = 2

# Load IMDb dataset
print("Loading IMDb dataset...")
dataset, info = tfds.load("imdb_reviews", split=["train", "test"], as_supervised=True, with_info=True)
train_data, test_data = dataset[0], dataset[1]
print("Dataset loaded.")

print("Processing training data...")


binary_embeddings = []
labels = []

for i in range(batch_numbers):
    texts = []
    num = i*batch_size
    print("getting", len(train_data.skip(num).take(batch_size)), "samples from", num, "to", num + batch_size)
    for text, label in train_data.skip(num).take(batch_size):
        texts.append(clean_text(text.numpy().decode('utf-8')))
        labels.append([int(label.numpy())])
        
    # Get embeddings and convert them to binary
    print("grabbing embeddings")
    embeddings = be.get_embedding_batch(texts)

    print("converting to binary")

    binary_embeddings_indi = (be.embeddingsToBinaryBatch(embeddings))
    for binary_embedding in binary_embeddings_indi:
        binary_embeddings.append(binary_embedding)





print("finished processing training data.")

print("Training agent...")


print("Training agent on", len(binary_embeddings), "samples")
# Train the agent on the batch


agent.next_state_batch(INPUT=binary_embeddings, LABEL=labels, unsequenced=True, print_result=True)

print("Training complete. Starting testing...")

# Process test data
test_texts, test_labels, test_embeddings = [], [], []

for text, label in test_data.take(500):
    decoded_text = clean_text(text.numpy().decode('utf-8'))
    test_texts.append(decoded_text)
    test_labels.append(label.numpy())


test_embeddings = be.get_embedding_batch(test_texts)

test_embeddings_binary = []
for i in range(len(test_embeddings)):
    test_embeddings_binary.append(be.embeddingToBinary(test_embeddings[i]))

# Evaluate agent
success = 0
for i, binary_embedding in enumerate(test_embeddings_binary):
    response = agent.next_state(binary_embedding, unsequenced=True)
    agent.reset_state()
    prediction = 1 if sum(response) >=9 else 0

    print(f"Test: {test_texts[i]}\nPredicted: {prediction}, Actual: {test_labels[i ]}\n")
    
    if test_labels[i] == prediction:
        success += 1

print(f"Success rate: {success}/{len(test_embeddings)} ({(success / len(test_embeddings)) * 100:.2f}%)")

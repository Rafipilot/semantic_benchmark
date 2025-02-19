import tensorflow_datasets as tfds
from openai import OpenAI
import ao_embeddings.binaryEmbeddings as be
import ao_core as ao
import ao_arch as ar
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ.get("openai_api_key")

arch_i, arch_z, arch_c = [1536], [10], [0]
connector_function = "full_conn"
Arch = ar.Arch(arch_i, arch_z, arch_c, connector_function, description="none")
agent = ao.Agent(Arch)
agent.full_conn_compress = True

client = OpenAI(api_key=api_key)
be = be.binaryEmbeddings(api_key, cacheName="cache.json", numberBinaryDigits=1536)


batch_size = 2000

# Load IMDb dataset
dataset, info = tfds.load("imdb_reviews", split=["train", "test"], as_supervised=True, with_info=True)
train_data, test_data = dataset[0], dataset[1]


texts, labels = [], []

for text, label in train_data.take(batch_size):
    texts.append(text.numpy().decode('utf-8'))
    labels.append([int(label.numpy())])

embeddings = []


embeddings = be.get_embedding_batch(texts)

# Convert to binary embeddings
binary_embeddings = []

for embedding in embeddings:
    binary_embedding = be.embeddingToBinary(embedding)
    binary_embeddings.append(binary_embedding)


agent.next_state_batch(INPUT=binary_embeddings, LABEL=labels, unsequenced=True, DD=False, print_result=True)

print("Training complete. Starting testing...")

# Process test data
test_texts, test_labels, test_embeddings = [], [], []

for text, label in test_data.take(100):
    decoded_text = text.numpy().decode('utf-8')
    test_texts.append(decoded_text)
    test_labels.append(label.numpy())

print(test_texts[3])
test_embeddings = be.get_embedding_batch(test_texts)
print(test_embeddings[3])
test_embeddings_binary = []
for i in range(len(test_embeddings)):
    test_embeddings_binary.append(be.embeddingToBinary(test_embeddings[i]))

# Evaluate agent
success = 0
for i, binary_embedding in enumerate(test_embeddings_binary):
    response = agent.next_state(binary_embedding, DD=False, unsequenced=True)
    agent.reset_state()
    prediction = 1 if sum(response) >= 5 else 0

    print(f"Test: {test_texts[i]}\nPredicted: {prediction}, Actual: {test_labels[i]}\n")
    
    if test_labels[i] == prediction:
        success += 1

print(f"Success rate: {success}/{len(test_embeddings)} ({(success / len(test_embeddings)) * 100:.2f}%)")

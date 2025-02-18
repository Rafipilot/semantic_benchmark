
import tensorflow_datasets as tfds
from openai import OpenAI

import ao_embeddings.binaryEmbeddings as be
import ao_core as ao
import ao_arch as ar

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ.get("openai_api_key")

arch_i = [128]
arch_z = [1]
arch_c = [0]
connector_function = "full_conn"

Arch = ar.Arch(arch_i, arch_z, arch_c, connector_function, description="none")

agent = ao.Agent(Arch)

client = OpenAI(api_key = api_key,)

be = be.binaryEmbeddings(api_key, cacheName="cache.json", numberBinaryDigits=128 )

# Load IMDb dataset
dataset, info = tfds.load("imdb_reviews", split=["train", "test"], as_supervised=True, with_info=True)

# Split into training and testing sets
train_data, test_data = dataset[0], dataset[1]


texts = []
labels = []

# Check dataset structure
for text, label in train_data.take(1000):
    text = text.numpy().decode('utf-8')  # Decode tensor to string
    texts.append(text)
    label = [int(label.numpy())]
    labels.append(label)

response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=texts
)

embeddings = [item.embedding for item in response.data]


binary_embeddings = []

for i in range(len(embeddings)):
    binary_embedding = be.embeddingToBinary(embeddings[i])
    binary_embeddings.append(binary_embedding)

agent.next_state_batch(
    INPUT=binary_embeddings,
    LABEL=labels,
    unsequenced = True,
    DD=False,
    print_result=True
)
print("done train start test")

test_embeddings = []
test = []
test_text = []

for text, label in test_data.take(20):
    test_text.append(text.numpy().decode('utf-8'))
    embedding = be.get_embedding(text.numpy().decode('utf-8'))
    binary_embedding = be.embeddingToBinary(embedding)
    test_embeddings.append(binary_embedding)
    test.append([test])


print("done")

for i in range(len(test_embeddings)):
    response = agent.next_state(test_embeddings[i], DD=False, unsequenced = True)
    agent.reset_state()
    print(f"test: {test_text[i]} ", response)


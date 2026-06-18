from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# 1. Load embeddings (same model as before)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings
)

retriever = db.as_retriever()

# 3. FREE local LLM (lightweight)
pipe = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=200
)

llm = HuggingFacePipeline(pipeline=pipe)

# 4. RAG chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

print("Chatbot ready! Type 'exit' to stop\n")

# 5. Chat loop
while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    response = qa.run(query)
    print("\nBot:", response, "\n")
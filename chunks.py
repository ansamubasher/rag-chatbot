from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# 1. Load PDFs
loader = PyPDFDirectoryLoader("pdfs")
documents = loader.load()

print("Pages loaded:", len(documents))

# 2. Create chunker
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # size of each chunk
    chunk_overlap=100    # overlap between chunks
)

# 3. Split documents into chunks
chunks = splitter.split_documents(documents)

# 4. Results
print("Total chunks created:", len(chunks))

# 5. Show sample chunk
print("\n--- SAMPLE CHUNK ---\n")
print(chunks[0].page_content)


#################

# 3. FREE embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="vectorstore"
)

vectorstore.persist()
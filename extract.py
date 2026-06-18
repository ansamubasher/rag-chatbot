from langchain_community.document_loaders import PyPDFDirectoryLoader

# STEP 1: Load all PDFs from folder
loader = PyPDFDirectoryLoader("pdfs")
documents = loader.load()

# STEP 2: Print basic info
print("Total pages loaded:", len(documents))
print("\n--- SAMPLE TEXT ---\n")

# STEP 3: Show first document content
print(documents[0].page_content[:1000])
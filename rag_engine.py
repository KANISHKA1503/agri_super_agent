import os
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Lazily initialize the embedding model so it doesn't block server startup
embedding_function = None

def get_embedding_function():
    global embedding_function
    if embedding_function is None:
        print("[RAG] Initializing HuggingFaceEmbeddings (this might take a moment)...")
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedding_function

# Define where to save the database
DB_DIR = "./vector_db"


def clean_text(text):
    """Removes extra whitespace, newlines, and handles NaN values from CSV data."""
    if pd.isna(text):
        return ""
    return " ".join(str(text).split())


def add_documents_in_batches(docs, collection_name, batch_size=5000):
    """
    Inserts documents into ChromaDB in safe batches to avoid
    SQLite variable limits and memory issues with large datasets.
    """
    if not docs:
        return
    total_docs = len(docs)
    total_batches = ((total_docs - 1) // batch_size) + 1
    print(f"  Indexing {total_docs} documents into '{collection_name}' in {total_batches} batches...")

    # Create the collection with the first batch
    first_batch = docs[:batch_size]
    db = Chroma.from_documents(
        first_batch,
        get_embedding_function(),
        persist_directory=DB_DIR,
        collection_name=collection_name
    )
    print(f"  -> Batch 1/{total_batches} done ({len(first_batch)} docs)")

    # Add subsequent batches
    for i in range(batch_size, total_docs, batch_size):
        batch = docs[i : i + batch_size]
        db.add_documents(batch)
        batch_num = (i // batch_size) + 1
        print(f"  -> Batch {batch_num}/{total_batches} done ({min(i + batch_size, total_docs)}/{total_docs} docs)")


def build_knowledge_databases():
    """
    Reads BOTH CSVs and builds two separate local Chroma vector databases.
    Run this ONCE to create the database folder.
    """
    import shutil

    print("--- Building Databases ---")

    # Clear existing DB to prevent duplicates on rebuild
    if os.path.exists(DB_DIR):
        print(f"Clearing existing vector database at {DB_DIR}...")
        try:
            shutil.rmtree(DB_DIR)
        except Exception as e:
            print(f"Warning: Could not clear {DB_DIR}: {e}")

    # ---------------------------------------------------------
    # 1. Build General Knowledge DB (questionsv4.csv)
    # ---------------------------------------------------------
    file1 = "data/questionsv4.csv"
    if os.path.exists(file1):
        print(f"\nProcessing {file1}...")
        df1 = pd.read_csv(file1).dropna(subset=['questions', 'answers'])
        docs1 = []
        for _, row in df1.iterrows():
            q = clean_text(row['questions'])
            a = clean_text(row['answers'])
            content = f"Farmer Issue: {q}\nExpert Solution: {a}"
            docs1.append(Document(page_content=content, metadata={"source": "general_knowledge"}))

        add_documents_in_batches(docs1, "general_knowledge", batch_size=5000)
        print(f"[OK] General Knowledge DB built with {len(docs1)} records.")
    else:
        print(f"[WARNING] {file1} not found. Skipping General DB.")

    # ---------------------------------------------------------
    # 2. Build Disease DB (FarmGenie CSV)
    # ---------------------------------------------------------
    file2 = "data/qna-dataset-farmgenie-plant-diseases_v2.csv"
    if os.path.exists(file2):
        print(f"\nProcessing {file2}...")
        df2 = pd.read_csv(file2).dropna(subset=['QUESTION.question', 'ANSWER'])
        docs2 = []
        for _, row in df2.iterrows():
            q = clean_text(row['QUESTION.question'])
            a = clean_text(row['ANSWER'])
            content = f"Farmer Issue: {q}\nExpert Solution: {a}"
            docs2.append(Document(page_content=content, metadata={"source": "disease_knowledge"}))

        add_documents_in_batches(docs2, "disease_knowledge", batch_size=5000)
        print(f"[OK] Disease DB built with {len(docs2)} records.")
    else:
        print(f"[WARNING] {file2} not found. Skipping Disease DB.")


def retrieve_context(user_query: str, collection_name: str, k: int = 3) -> str:
    """
    Searches the SPECIFIC database requested by the Router.
    k=3 is optimal for voice responses (concise context = shorter TTS output).
    """
    try:
        db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=get_embedding_function(),
            collection_name=collection_name
        )

        results = db.similarity_search(user_query, k=k)

        if not results:
            return "No specific agricultural data found for this query in the offline database."

        return "\n\n---\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        return f"Database error: {str(e)}"


# ==========================================
# TEST THE ENGINE (Run this file directly)
# ==========================================
if __name__ == "__main__":

    # STEP 1: Build the DB (Keep this commented out unless you need to rebuild the vector_db folder)
    # build_knowledge_databases()

    # STEP 2: Test the retrieval on BOTH databases
    print("\n--- Testing RAG Retrieval ---")

    query1 = "How can I get a loan for my farm?"
    print(f"\n[Testing General DB] Query: {query1}")
    print(retrieve_context(query1, collection_name="general_knowledge"))

    query2 = "What is the root feed recommendation for managing tree diseases?"
    print(f"\n[Testing Disease DB] Query: {query2}")
    print(retrieve_context(query2, collection_name="disease_knowledge"))
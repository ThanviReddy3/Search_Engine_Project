import streamlit as st
import pandas as pd
from tqdm import tqdm
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Define global variables
model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_size = 3000
overlap_size = 100
chroma_client = PersistentClient(path="my_vectordb")
model_name = "all-mpnet-base-v2"
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
collection_name = "eng_subtitles_collection"


def chunk_embeddings_with_overlap(embeddings, chunk_size, overlap_size, chunk_ids, metadata, documents):
    chunked_embeddings = []
    chunked_ids = []
    chunked_metadata = []
    chunked_documents = []
    for idx in range(0, len(embeddings), chunk_size - overlap_size):
        chunk = embeddings[idx:idx + chunk_size]  # Extract chunk with overlap
        chunked_embeddings.append(chunk)
        chunked_ids.append(chunk_ids[idx:idx + chunk_size])  # Match IDs to the corresponding chunk
        chunked_metadata.append(metadata[idx:idx + chunk_size])  # Match metadata to the corresponding chunk
        chunked_documents.append(documents[idx:idx + chunk_size])  # Match documents to the corresponding chunk
    return pd.DataFrame({'chunk_id': chunked_ids, 'embedding': chunked_embeddings,
                         'metadata': chunked_metadata, 'documents': chunked_documents})


def search(query, collection):
    query_embedding = model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=10,
        include=['documents', 'distances', 'metadatas']
    )

    if results and 'metadatas' in results and results['metadatas']:
        for j in range(len(results['metadatas'][0])):
            movie_id = results["ids"][0][j]
            distance = results['distances'][0][j]
            metadata = results['metadatas'][0][j]

            if metadata and 'name' in metadata:
                st.write(f"Movie Name: {metadata['name']}")
                st.write(f"Similarity Score (distance): {distance:.3f}")

            else:
                st.write("Metadata for this result is missing or invalid.")
    else:
        st.write("No results found or metadata is missing in the results.")


def main():
    st.title("Optimizing Semantic Search for Movie Subtitles")

    # Load data and prepare collection
    df = pd.read_csv("half_subs_clean.csv", nrows=5000)
    df['name'] = df['name'].str.replace('.', ' ').str.replace('eng 1cd', '').str.title().str.strip()
    ids = [str(i) for i in range(1, len(df) + 1)]
    df['id'] = ids
    df['id'] = df['id'].astype('int64')
    embeddings = model.encode(df['file_content'])

    # Prepare metadata and documents
    metadata = df[['id', 'name']].to_dict(orient='records')
    documents = df['file_content'].tolist()

    # Chunk embeddings with overlap
    chunked_data = chunk_embeddings_with_overlap(embeddings, chunk_size, overlap_size, ids, metadata, documents)

    try:
        collection = chroma_client.get_collection(name=collection_name, embedding_function=sentence_transformer_ef)
    except ValueError:
        st.write(f"Collection '{collection_name}' does not exist. Creating a new collection.")
        collection = chroma_client.create_collection(name=collection_name, embedding_function=sentence_transformer_ef)


    with tqdm(total=len(chunked_data), desc="Progress", unit="chunk", ncols=100) as pbar:
        # Add embeddings in batches
        for chunk_id, chunk_embedding, chunk_metadata, chunk_documents in zip(chunked_data['chunk_id'],
                                                                              chunked_data['embedding'],
                                                                              chunked_data['metadata'],
                                                                              chunked_data['documents']):
            collection.add(
                documents=chunk_documents,
                embeddings=chunk_embedding,
                metadatas=chunk_metadata,
                ids=chunk_id
            )
            pbar.update(1)  # Update progress for each chunk


    user_query = st.text_input("Enter the keyword:")
    if st.button("Search"):
        search(user_query, collection)


if __name__ == "__main__":
    main()



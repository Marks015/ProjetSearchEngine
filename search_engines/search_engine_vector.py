import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch


class DenseVectorSearchEngine:
    def __init__(self, data_path, queries_path, model_name="multi-qa-mpnet-base-dot-v1", top_k=10, batch_size=32):
        self.data_path = data_path
        self.queries_path = queries_path
        self.top_k = top_k
        self.batch_size = batch_size
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")

        # Load multilingual model
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.documents = {}  # Map doc_id to content
        self.doc_ids = []  # Keep track of document IDs
        self.embeddings_file = "search_engines/document_embeddings.pckl"

        # Load and encode documents
        self.document_embeddings = self.load_or_encode_documents()

    def load_or_encode_documents(self):
        """Load document embeddings from pickle if available, otherwise encode and save."""
        if os.path.exists(self.embeddings_file):
            print("Loading precomputed embeddings from pickle...")
            with open(self.embeddings_file, "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.doc_ids = data["doc_ids"]
                return torch.tensor(data["embeddings"]).to(self.device)

        print("No pickle file found. Encoding documents...")
        return self.encode_and_save_documents()

    def encode_and_save_documents(self):
        """Encode documents and save embeddings to a pickle file."""
        doc_texts = []

        # Read each document in the data path
        for doc_file in os.listdir(self.data_path):
            doc_id = doc_file.split(".txt")[0]
            with open(os.path.join(self.data_path, doc_file), 'r', encoding='utf-8') as f:
                content = f.read().strip()
                self.documents[doc_id] = content
                doc_texts.append(content)
                self.doc_ids.append(doc_id)

        # Encode documents in batches to avoid memory issues
        print("Encoding documents in batches...")
        all_embeddings = []
        for i in range(0, len(doc_texts), self.batch_size):
            batch_texts = doc_texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(batch_texts, convert_to_tensor=True).to(self.device)
            all_embeddings.append(batch_embeddings)

        # Concatenate all batch embeddings
        embeddings = torch.cat(all_embeddings, dim=0)

        # Save embeddings to a pickle file
        with open(self.embeddings_file, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "doc_ids": self.doc_ids,
                "embeddings": embeddings.cpu().numpy()
            }, f)

        print("Embeddings saved to pickle file.")
        return embeddings

    def search(self, query):
        """Search for the top K documents that are most similar to the query."""
        query_embedding = self.model.encode(query, convert_to_tensor=True).to(self.device)
        similarity_scores = util.pytorch_cos_sim(query_embedding, self.document_embeddings)[0]

        # Apply Length Normalization and Dynamic Keyword Boosting
        query_keywords = set(query.split())
        ranked_results = []
        for i, score in enumerate(similarity_scores):
            doc_id = self.doc_ids[i]
            doc_length = len(self.documents[doc_id].split())
            normalized_score = score / (np.log(doc_length + 1) + 1)

            # Dynamic keyword boosting
            document_text = self.documents[doc_id].lower()
            for keyword in query_keywords:
                if keyword in document_text:
                    normalized_score += 0.1
                    break

            ranked_results.append((doc_id, normalized_score.item()))

        ranked_results = sorted(ranked_results, key=lambda x: x[1], reverse=True)[:self.top_k]
        return ranked_results

    def evaluate(self):
        """Evaluate accuracy, precision, recall, and MRR; save top 5 results for each query."""
        total_queries = 0
        correct_matches = 0
        precision_sum = 0
        recall_sum = 0
        reciprocal_rank_sum = 0

        query_results = {}

        # Open and process the JSONL file
        with open(self.queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Each line in the file is a separate JSON object
                query_data = json.loads(line)
                queries = query_data["Queries"]
                answer_file = query_data["Answer file"].split(".txt")[0]

                for query in queries:
                    total_queries += 1
                    top_results = self.search(query)
                    retrieved_docs = [doc_id for doc_id, _ in top_results]

                    # Save the top 5 results for this query
                    query_results[query] = retrieved_docs[:5]

                    # Calculate accuracy, precision, recall, MRR
                    if answer_file in retrieved_docs:
                        correct_matches += 1
                        rank = retrieved_docs.index(answer_file) + 1
                        reciprocal_rank_sum += 1 / rank

                    relevant_retrieved = 1 if answer_file in retrieved_docs else 0
                    precision = relevant_retrieved / len(retrieved_docs)
                    recall = relevant_retrieved

                    precision_sum += precision
                    recall_sum += recall

        # Calculate average metrics
        accuracy = correct_matches / total_queries if total_queries > 0 else 0
        avg_precision = precision_sum / total_queries if total_queries > 0 else 0
        avg_recall = recall_sum / total_queries if total_queries > 0 else 0
        mrr = reciprocal_rank_sum / total_queries if total_queries > 0 else 0

        # Output metrics
        print(f"\nTotal queries: {total_queries}")
        print(f"Correct matches: {correct_matches}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Avg Precision: {avg_precision:.4f}")
        print(f"Avg Recall: {avg_recall:.4f}")
        print(f"MRR: {mrr:.4f}")

import os
import math
import json
from collections import defaultdict, Counter
from utils.utils import preprocess_text  # Updated to include new preprocessing functions
import numpy as np

class TFIDFSearchEngine:
    def __init__(self, data_path, queries_path):
        self.data_path = data_path
        self.queries_path = queries_path
        self.index = defaultdict(list)  # Reverse index
        self.doc_count = defaultdict(int)  # To store document frequencies (for IDF)
        self.documents = {}  # Map doc_id to the content of the document
        self.total_docs = 0  # Total number of documents in the corpus
        self.load_documents()
    
    def load_documents(self):
        """Load all documents and prepare the reverse index."""
        for doc_file in os.listdir(self.data_path):
            doc_id = doc_file.split(".txt")[0]
            with open(os.path.join(self.data_path, doc_file), 'r', encoding='utf-8') as f:
                content = f.read()
                processed_content = preprocess_text(content)
                self.documents[doc_id] = processed_content
                self.index_document(doc_id, processed_content)
        self.total_docs = len(self.documents)
    
    def index_document(self, doc_id, document):
        """Create reverse index and calculate term frequencies with sublinear scaling."""
        terms = document.split()
        term_frequencies = Counter(terms)
        
        for term, count in term_frequencies.items():
            # Sublinear TF scaling (1 + log(TF))
            tf = 1 + math.log(count)
            self.index[term].append((doc_id, tf))  # Store scaled TF in the index
            self.doc_count[term] += 1  # Increment document frequency of the term
    
    def compute_idf(self, term):
        """Compute the inverse document frequency (IDF) of a term."""
        # Adjust IDF to minimize the effect of very frequent terms
        return math.log((self.total_docs + 1) / (1 + self.doc_count[term])) + 1
    
    def compute_tf_idf(self, terms):
        """Compute TF-IDF vector for a query or document with normalization."""
        tf_idf_vector = defaultdict(float)
        term_frequencies = Counter(terms)
        total_terms = len(terms)
        
        for term, count in term_frequencies.items():
            # Sublinear TF scaling
            tf = 1 + math.log(count)
            idf = self.compute_idf(term) if term in self.doc_count else 0
            tf_idf_vector[term] = (tf * idf) / total_terms  # Normalize by document length
        
        # Normalize the vector for better cosine similarity calculation
        norm = math.sqrt(sum(val ** 2 for val in tf_idf_vector.values()))
        if norm > 0:
            for term in tf_idf_vector:
                tf_idf_vector[term] /= norm
        
        return tf_idf_vector
    
    def cosine_similarity(self, vec1, vec2):
        """Compute the cosine similarity between two TF-IDF vectors."""
        dot_product = sum(vec1[term] * vec2.get(term, 0) for term in vec1)
        magnitude1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
        magnitude2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def search(self, query, top_k=10):
        """Search for documents that match the query based on TF-IDF and cosine similarity."""
        # Preprocess the query to improve matching
        query_terms = preprocess_text(query).split()
        query_tf_idf = self.compute_tf_idf(query_terms)
        
        doc_scores = []
        for doc_id, document in self.documents.items():
            doc_terms = document.split()
            doc_tf_idf = self.compute_tf_idf(doc_terms)
            similarity = self.cosine_similarity(query_tf_idf, doc_tf_idf)
            doc_scores.append((doc_id, similarity))
        
        # Sort by similarity and return only the top_k documents
        return sorted(doc_scores, key=lambda x: x[1], reverse=True)[:top_k]
    
    def run_queries(self):
        """Run all the queries and return the best matching document for each."""
        correct_matches = 0  # To count how many queries return the correct document
        total_queries = 0  # Total number of queries processed
        
        with open(self.queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                query_data = json.loads(line)
                queries = query_data["Queries"]
                answer_file = query_data["Answer file"].split(".txt")[0]  # Remove ".txt" to match doc_id
                total_queries += len(queries)
                
                for query in queries:
                    results = self.search(query)
                    best_match = results[0][0] if results else None
                    
                    if best_match == answer_file:
                        correct_matches += 1
                    
                    print(f"Query: {query} | Best Match: {best_match} | Expected: {answer_file}")
        
        # Output final count of correct queries
        print(f"\nTotal queries: {total_queries}")
        print(f"Correct matches: {correct_matches}")
        print(f"Accuracy: {correct_matches / total_queries * 100:.2f}%")

    def evaluate(self):
        """Evaluate the search engine using precision, recall, MRR, and accuracy."""
        precision_sum = 0
        recall_sum = 0
        reciprocal_rank_sum = 0
        correct_matches = 0  # Count of correct top-ranked matches
        total_queries = 0

        with open(self.queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                query_data = json.loads(line)
                queries = query_data["Queries"]
                answer_file = query_data["Answer file"].split(".txt")[0]  # Ground truth relevant document

                total_queries += len(queries)

                for query in queries:
                    # Search for the relevant documents for this query
                    results = self.search(query)
                    
                    # Retrieve document IDs of top-ranked results
                    retrieved_docs = [doc[0] for doc in results]  # Top K retrieved documents
                    relevant_docs = [answer_file]  # There's typically one relevant doc per query

                    # True Positives = Relevant documents retrieved by the search engine
                    relevant_retrieved = [doc for doc in retrieved_docs if doc in relevant_docs]

                    # Calculate precision: How many of the retrieved documents are relevant?
                    precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0

                    # Calculate recall: How many of the relevant documents were retrieved?
                    recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0

                    # Sum the precision and recall for averaging later
                    precision_sum += precision
                    recall_sum += recall

                    # Calculate Reciprocal Rank (for MRR): Rank of the first relevant document
                    for rank, doc in enumerate(retrieved_docs, start=1):
                        if doc == answer_file:
                            reciprocal_rank_sum += 1 / rank
                            break  # Stop once the first relevant document is found
                    
                    # Check if the top-ranked document is the correct answer
                    if retrieved_docs and retrieved_docs[0] == answer_file:
                        correct_matches += 1

        # Calculate average metrics
        avg_precision = precision_sum / total_queries
        avg_recall = recall_sum / total_queries
        mrr = reciprocal_rank_sum / total_queries
        accuracy = correct_matches / total_queries  # Calculate accuracy

        # Output the evaluation results
        print(f"Avg Precision: {avg_precision:.4f}")
        print(f"Avg Recall: {avg_recall:.4f}")
        print(f"MRR: {mrr:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

import json
from search_engines.search_engine_tfidf import TFIDFSearchEngine  # Replace with the actual TF-IDF class
from search_engines.search_engine_vector import DenseVectorSearchEngine  # Replace with the actual Dense Vector class


class HybridSearchEngine:
    def __init__(self, tfidf_engine, dense_engine, top_k=10):
        self.tfidf_engine = tfidf_engine
        self.dense_engine = dense_engine
        self.top_k = top_k

    def search(self, query):
        """Combine TF-IDF and Dense Vector results by summing the scores."""
        # Get results from both engines
        tfidf_results = self.tfidf_engine.search(query, top_k=self.top_k)
        dense_results = self.dense_engine.search(query)

        # Create a dictionary to combine scores
        combined_scores = {}

        # Aggregate TF-IDF scores
        for doc_id, score in tfidf_results:
            combined_scores[doc_id] = score

        # Aggregate Dense Vector scores
        for doc_id, score in dense_results:
            if doc_id in combined_scores:
                combined_scores[doc_id] += score
            else:
                combined_scores[doc_id] = score

        # Sort by combined score and return top K
        ranked_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        return ranked_results

    def evaluate(self, queries_path, output_file="hybrid_results.json"):
        """Evaluate the hybrid search engine and save results."""
        total_queries = 0
        correct_matches = 0
        precision_sum = 0
        recall_sum = 0
        reciprocal_rank_sum = 0

        query_results = {}

        # Load queries and evaluate
        with open(queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                query_data = json.loads(line)
                queries = query_data["Queries"]
                answer_file = query_data["Answer file"].split(".txt")[0]  # Ground truth document ID

                for query in queries:
                    total_queries += 1
                    top_results = self.search(query)
                    retrieved_docs = [doc_id for doc_id, _ in top_results]

                    # Save the top 5 results for each query
                    query_results[query] = retrieved_docs[:5]

                    # Calculate accuracy, precision, recall, and MRR
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

        # Save results
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(query_results, outfile, ensure_ascii=False, indent=4)

        # Output metrics
        print(f"\nTotal queries: {total_queries}")
        print(f"Correct matches: {correct_matches}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Avg Precision: {avg_precision:.4f}")
        print(f"Avg Recall: {avg_recall:.4f}")
        print(f"MRR: {mrr:.4f}")


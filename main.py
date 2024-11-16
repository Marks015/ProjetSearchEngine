from search_engines.search_engine_tfidf import TFIDFSearchEngine
from search_engines.search_engine_vector import DenseVectorSearchEngine
from search_engines.hybrid_search_engine import HybridSearchEngine
def main():
    # Define paths
    data_path = "data/wiki_split_extract_2k"  # Path to the folder containing document files
    queries_path = "data/requetes.jsonl"  # Path to the queries JSONL file

    # Evaluate TF-IDF Search Engine
    print("\n--- Evaluating TF-IDF Search Engine ---")
    tfidf_engine = TFIDFSearchEngine(data_path, queries_path)
    tfidf_engine.evaluate()

    # Evaluate Dense Vector Search Engine
    print("\n--- Evaluating Dense Vector Search Engine ---")
    vector_engine = DenseVectorSearchEngine(data_path,queries_path)
    vector_engine.evaluate()

    #Evaluate Hybrid Search Engine
    print("\n--- Evaluating Hybrid  Search Engine ---")
    hybrid_engine=HybridSearchEngine(tfidf_engine,vector_engine)
    hybrid_engine.evaluate(queries_path)


if __name__ == "__main__":
    main()

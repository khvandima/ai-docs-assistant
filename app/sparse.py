from fastembed import SparseTextEmbedding

# Единственный экземпляр BM25 модели.
# Импортируется и в agent.py и в ingestion.py — инициализируется один раз.
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
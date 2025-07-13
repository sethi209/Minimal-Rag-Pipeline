from app import retriever

def test_retrieval_returns_documents():
    results = retriever.vectorstore.similarity_search_with_score("transformer", k=3)
    assert len(results) == 3
    for doc, score in results:
        assert isinstance(doc.page_content, str)
        assert 0 <= score <= 1

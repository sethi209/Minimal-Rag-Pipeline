from app import retrieval_chain

def test_rag_chain():
    response = retrieval_chain.invoke({"input": "What is a transformer model?"})
    assert "answer" in response
    assert isinstance(response["answer"], str)

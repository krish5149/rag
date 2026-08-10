from sentence_transformers import CrossEncoder

class ReRanker:

    def __init__(self,reranker_model):

        self.reranker = CrossEncoder(reranker_model)


    def rerank(self,documents,question: str):

        # extract the content from document
        docs_content = [doc.page_content for doc in documents]

        # pair the each content with the question
        pair = [[question,content] for content in docs_content]

        # score for each pair
        scores = self.reranker.predict(pair)

        # zip the content and score alone
        scored_docs = list(zip(docs_content,scores))

        # sort the scored_docs
        sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

        return sorted_docs[:10]
        

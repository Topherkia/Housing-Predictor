from pathlib import Path

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


class HousingRAG:

    def __init__(
        self,
        knowledge_dir="knowledge_base"
    ):

        self.knowledge_dir = Path(knowledge_dir)

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.documents = []
        self.index = None

        self.load_documents()
        self.build_index()

    def load_documents(self):

        for file in self.knowledge_dir.glob("*.md"):

            text = file.read_text(
                encoding="utf-8"
            )

            self.documents.append({
                "source": file.name,
                "text": text
            })

    def build_index(self):

        texts = [
            document["text"]
            for document in self.documents
        ]

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True
        )

        embeddings = embeddings.astype(
            np.float32
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.index.add(embeddings)

    def retrieve(self, query, k=3):

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )

        query_embedding = query_embedding.astype(
            np.float32
        )

        distances, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        for index in indices[0]:

            if index < len(self.documents):

                results.append(
                    self.documents[index]
                )

        return results
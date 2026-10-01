from pathlib import Path

import faiss
import numpy as np

from sentence_transformers import (
    SentenceTransformer,
)


class HousingRAG:

    def __init__(
        self,
        knowledge_dir="knowledge_base",
        embedding_model=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
    ):

        self.knowledge_dir = Path(
            knowledge_dir
        )

        self.embedding_model = (
            SentenceTransformer(
                embedding_model
            )
        )

        self.documents = []
        self.index = None

        self._load_documents()
        self._build_index()

    def _load_documents(self):

        self.documents = []

        for path in sorted(
            self.knowledge_dir.glob(
                "*.md"
            )
        ):

            text = path.read_text(
                encoding="utf-8"
            ).strip()

            if not text:
                continue

            # Split documents into chunks.
            chunks = [
                chunk.strip()
                for chunk in text.split(
                    "\n\n"
                )
                if chunk.strip()
            ]

            for chunk in chunks:

                self.documents.append(
                    {
                        "source":
                            path.name,

                        "text":
                            chunk,
                    }
                )

        if not self.documents:

            raise FileNotFoundError(
                "No knowledge-base Markdown "
                f"files found in "
                f"{self.knowledge_dir}"
            )

    def _build_index(self):

        texts = [
            doc["text"]
            for doc in self.documents
        ]

        embeddings = (
            self.embedding_model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            .astype(
                np.float32
            )
        )

        dimension = (
            embeddings.shape[1]
        )

        self.index = (
            faiss.IndexFlatIP(
                dimension
            )
        )

        self.index.add(
            embeddings
        )

    def retrieve(
        self,
        query,
        k=4,
    ):

        query_embedding = (
            self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            .astype(
                np.float32
            )
        )

        scores, indices = (
            self.index.search(
                query_embedding,
                min(
                    k,
                    len(
                        self.documents
                    ),
                ),
            )
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            if index < 0:
                continue

            document = dict(
                self.documents[index]
            )

            document["score"] = float(
                score
            )

            results.append(
                document
            )

        return results
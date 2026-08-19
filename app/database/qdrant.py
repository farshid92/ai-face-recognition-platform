from qdrant_client import QdrantClient, models
from uuid import uuid4
from dataclasses import dataclass
import numpy as np

@dataclass
class FaceMatch:
    name: str
    score: float

class FaceDatabase:
    """Manage face embeddings stored in Qdrant."""

    COLLECTION_NAME = "registered_faces"
    VECTOR_SIZE = 512

    def __init__(
        self,
        url: str = "http://localhost:6333",
    ) -> None:
        self.client = QdrantClient(url=url)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections()

        exists = any(
            collection.name == self.COLLECTION_NAME
            for collection in collections.collections
        )

        if not exists:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )

    def register(
        self,
        name: str,
        embedding: np.ndarray,
    ) -> str:
        point_id = str(uuid4())

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            wait=True,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "name": name,
                    },
                )
            ],
        )

        return point_id

    def search(
        self,
        embedding: np.ndarray,
        limit: int = 1,
    ):
        return self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=embedding.tolist(),
            limit=limit,
        ).points

    def find_match(
        self,
        embedding: np.ndarray,
    ) -> FaceMatch | None:
        results = self.search(
            embedding,
            limit=3,
        )

        if not results:
            return None

        for result in results:
            print(
                f"Qdrant match: "
                f"{result.payload.get('name')} "
                f"score={result.score:.4f}"
            )

        result = results[0]

        return FaceMatch(
            name=result.payload["name"],
            score=float(result.score),
        )
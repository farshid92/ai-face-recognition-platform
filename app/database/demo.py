import numpy as np

from app.database.qdrant import FaceDatabase


def main() -> None:
    database = FaceDatabase()

    embedding_a = np.random.rand(512).astype(np.float32)
    embedding_a /= np.linalg.norm(embedding_a)

    embedding_b = np.random.rand(512).astype(np.float32)
    embedding_b /= np.linalg.norm(embedding_b)

    database.register(
        "Person A",
        embedding_a,
    )

    database.register(
        "Person B",
        embedding_b,
    )

    results = database.search(
        embedding_a,
        limit=2,
    )

    for result in results:
        print(
            f"Name: {result.payload['name']}"
        )
        print(
            f"Score: {result.score:.4f}"
        )


if __name__ == "__main__":
    main()
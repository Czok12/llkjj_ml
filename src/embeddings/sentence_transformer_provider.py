"""
Sentence Transformer Embedding Provider

Stateless Embedding Provider für das neue ML-Plugin Repository-Pattern.
Ersetzt die alte ResourceManager-basierte Implementierung.

Author: LLKJJ ML Team
Version: 1.0.0
Date: 2025-01-25
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class SentenceTransformerProvider:
    """
    Stateless Sentence Transformer Provider für Embeddings.

    Ersetzt ResourceManager-Singleton mit expliziter Resource-Verwaltung.
    Optimiert für deutsche Elektrohandwerk-Texte.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_folder: Path | None = None,
    ):
        """
        Initialize Sentence Transformer Provider.

        Args:
            model_name: Sentence-Transformers Model zu verwenden
            cache_folder: Optional custom cache directory
        """
        self.model_name = model_name
        self.cache_folder = cache_folder
        self._model = None  # Lazy loading

        logger.info(f"✅ SentenceTransformerProvider initialisiert: {model_name}")

    def encode(self, text: str) -> list[float]:
        """
        Encode Text zu Embedding-Vector.

        Args:
            text: Text zu encoden

        Returns:
            384-dimensionaler Float-Vector
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * 384

        try:
            # Lazy load model
            if self._model is None:
                self._load_model()

            # Encode text
            embedding = self._model.encode(text)

            # Convert to list and ensure 384 dimensions
            embedding_list = embedding.tolist()

            # Validate dimensions
            if len(embedding_list) != 384:
                logger.warning(
                    f"Unexpected embedding dimension: {len(embedding_list)}, expected 384"
                )
                # Pad or truncate to 384
                if len(embedding_list) < 384:
                    embedding_list.extend([0.0] * (384 - len(embedding_list)))
                else:
                    embedding_list = embedding_list[:384]

            return embedding_list

        except Exception as e:
            logger.error(f"Fehler bei Text-Encoding: {e}")
            # Return zero vector as fallback
            return [0.0] * 384

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Encode multiple texts in batch for better performance.

        Args:
            texts: Liste von Texten zu encoden

        Returns:
            Liste von 384-dimensionalen Float-Vectors
        """
        if not texts:
            return []

        try:
            # Lazy load model
            if self._model is None:
                self._load_model()

            # Encode batch
            embeddings = self._model.encode(texts)

            # Convert to list of lists
            embeddings_list = []
            for embedding in embeddings:
                embedding_list = embedding.tolist()

                # Validate and fix dimensions
                if len(embedding_list) != 384:
                    if len(embedding_list) < 384:
                        embedding_list.extend([0.0] * (384 - len(embedding_list)))
                    else:
                        embedding_list = embedding_list[:384]

                embeddings_list.append(embedding_list)

            return embeddings_list

        except Exception as e:
            logger.error(f"Fehler bei Batch-Encoding: {e}")
            # Return zero vectors as fallback
            return [[0.0] * 384 for _ in texts]

    def similarity(self, text1: str, text2: str) -> float:
        """
        Berechne Cosine-Similarity zwischen zwei Texten.

        Args:
            text1: Erster Text
            text2: Zweiter Text

        Returns:
            Cosine-Similarity (0.0 - 1.0)
        """
        try:
            embedding1 = np.array(self.encode(text1))
            embedding2 = np.array(self.encode(text2))

            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)

            # Normalize to 0-1 range (cosine similarity can be -1 to 1)
            return max(0.0, (similarity + 1) / 2)

        except Exception as e:
            logger.error(f"Fehler bei Similarity-Berechnung: {e}")
            return 0.0

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Model information dict
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": 384,
            "model_loaded": self._model is not None,
            "cache_folder": str(self.cache_folder) if self.cache_folder else None,
        }

    def _load_model(self) -> None:
        """Load Sentence Transformer model with proper error handling."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"🔄 Lade Sentence Transformer Model: {self.model_name}")

            # Load with custom cache folder if specified
            if self.cache_folder:
                self.cache_folder.mkdir(parents=True, exist_ok=True)
                self._model = SentenceTransformer(
                    self.model_name, cache_folder=str(self.cache_folder)
                )
            else:
                self._model = SentenceTransformer(self.model_name)

            logger.info(f"✅ Sentence Transformer Model geladen: {self.model_name}")

        except ImportError:
            raise RuntimeError(
                "sentence-transformers Package nicht installiert. "
                "Run: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(
                f"Fehler beim Laden des Sentence Transformer Models: {e}"
            )

    def cleanup(self) -> None:
        """Cleanup model resources."""
        if self._model is not None:
            # SentenceTransformer models don't need explicit cleanup
            # but we can clear the reference
            self._model = None
            logger.debug("✅ SentenceTransformer Model resources released")


# =============================================================================
# Factory Functions
# =============================================================================


def create_german_embedding_provider(
    cache_folder: Path | None = None,
) -> SentenceTransformerProvider:
    """
    Create optimized embedding provider for German text.

    Args:
        cache_folder: Optional cache directory

    Returns:
        Configured SentenceTransformerProvider for German
    """
    return SentenceTransformerProvider(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_folder=cache_folder,
    )


def create_multilingual_embedding_provider(
    cache_folder: Path | None = None,
) -> SentenceTransformerProvider:
    """
    Create general multilingual embedding provider.

    Args:
        cache_folder: Optional cache directory

    Returns:
        Configured SentenceTransformerProvider for multilingual use
    """
    return SentenceTransformerProvider(
        model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=cache_folder
    )


def create_embedding_provider_from_config(config: dict) -> SentenceTransformerProvider:
    """
    Create embedding provider from configuration dict.

    Args:
        config: Configuration with model_name and optional cache_folder

    Returns:
        Configured SentenceTransformerProvider
    """
    model_name = config.get(
        "model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    cache_folder = None

    if "cache_folder" in config and config["cache_folder"]:
        cache_folder = Path(config["cache_folder"])

    return SentenceTransformerProvider(model_name=model_name, cache_folder=cache_folder)

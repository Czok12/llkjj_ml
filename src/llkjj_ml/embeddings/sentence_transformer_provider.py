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
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    # Fallback type for when sentence_transformers is not available
    class SentenceTransformer:  # type: ignore[no-redef]
        pass


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
        eager_loading: bool = False,
    ):
        """
        Initialize Sentence Transformer Provider.

        Args:
            model_name: Sentence-Transformers Model zu verwenden
            cache_folder: Optional custom cache directory
            eager_loading: Load model immediately in constructor
        """
        self.model_name = model_name
        self.cache_folder = cache_folder
        self._model: SentenceTransformer | None = None  # Lazy loading

        if eager_loading:
            self._load_model()

        logger.info(f"✅ SentenceTransformerProvider initialisiert: {model_name}")

    def encode(
        self,
        texts: str | list[str],
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Encode text(s) to embeddings.

        Args:
            texts: Single text or list of texts
            normalize_embeddings: Whether to normalize embeddings
            convert_to_numpy: Convert to numpy array
            **kwargs: Additional arguments for sentence transformer
        """
        if self._model is None:
            self._load_model()

        # Handle empty input
        if not texts:
            if isinstance(texts, str):
                return np.array([0.0] * 384)
            else:
                return np.array([[0.0] * 384])

        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]

        try:
            assert self._model is not None, "Model should be loaded after _load_model()"
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=convert_to_numpy,
                **kwargs,
            )

            # Always return numpy array
            return np.array(embeddings)

        except Exception as e:
            logger.error(f"Fehler bei Text-Encoding: {e}")
            # Return zero vector as fallback
            if len(texts) == 1:
                return np.array([0.0] * 384)
            else:
                return np.array([[0.0] * 384 for _ in texts])

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = False,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """
        Encode batch of texts with custom batch size.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            show_progress_bar: Show progress bar
            normalize_embeddings: Whether to normalize embeddings
            **kwargs: Additional arguments
        """
        if not texts:
            return []

        if self._model is None:
            self._load_model()

        try:
            assert self._model is not None, "Model should be loaded after _load_model()"

            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                embeddings = self._model.encode(
                    batch,
                    normalize_embeddings=normalize_embeddings,
                    show_progress_bar=show_progress_bar and i == 0,
                    **kwargs,
                )
                # Ensure each batch result is an ndarray
                if isinstance(embeddings, np.ndarray):
                    # If model returns ndarray for whole batch, split rows
                    for row in np.atleast_2d(embeddings):
                        results.append(np.array(row))
                else:
                    results.extend([np.array(e) for e in embeddings])

            return results

        except Exception as e:
            logger.error(f"Fehler bei Batch-Encoding: {e}")
            # Return zero vectors as fallback
            return [np.array([0.0] * 384) for _ in texts]

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

            similarity = float(dot_product / (norm1 * norm2))

            # Normalize to 0-1 range (cosine similarity can be -1 to 1)
            return max(0.0, (similarity + 1) / 2)

        except Exception as e:
            logger.error(f"Fehler bei Similarity-Berechnung: {e}")
            return 0.0

    def get_model_info(self) -> dict[str, Any]:
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
            ) from None
        except Exception as e:
            raise RuntimeError(
                f"Fehler beim Laden des Sentence Transformer Models: {e}"
            ) from e

    def cleanup(self) -> None:
        """Cleanup model resources."""
        if self._model is not None:
            # SentenceTransformer models don't need explicit cleanup
            # but we can clear the reference
            self._model = None
            logger.debug("✅ SentenceTransformer Model resources released")


def similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Ensure 2D arrays
    if embedding1.ndim == 1:
        embedding1 = embedding1.reshape(1, -1)
    if embedding2.ndim == 1:
        embedding2 = embedding2.reshape(1, -1)

    return float(cosine_similarity(embedding1, embedding2)[0][0])


# Export functions
__all__ = ["SentenceTransformerProvider", "similarity"]


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


def create_embedding_provider_from_config(
    config: dict[str, Any],
) -> SentenceTransformerProvider:
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

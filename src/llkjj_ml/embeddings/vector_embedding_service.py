"""Vector Embedding Service for invoice similarity search and storage.

This service provides comprehensive embedding functionality with:
- German text processing using sentence-transformers
- pgvector for efficient similarity search
- SpaCy preprocessing and entity extraction
- Redis caching for performance
- HNSW indexing for fast approximate nearest neighbor search
"""

import logging
import re
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple

import psycopg2
import redis
import spacy
from sentence_transformers import SentenceTransformer

from llkjj_business.models.embedding_models import (
    DimensionError,
    EmbeddingResult,
    ModelLoadingError,
    SimilarityResult,
    StorageResult,
    UpdateResult,
)

logger = logging.getLogger(__name__)


class VectorEmbeddingService:
    """Service for creating, storing and searching vector embeddings of invoices."""

    VECTOR_DIMENSION = 768
    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        database_url: str,
        redis_url: str | None = None,
        spacy_model: str = "de_core_news_lg",
    ):
        """Initialize the VectorEmbeddingService.

        Args:
            database_url: PostgreSQL connection string with pgvector extension
            redis_url: Redis connection string for caching (optional)
            spacy_model: SpaCy model name for German text processing
        """
        self.database_url = database_url
        self.redis_url = redis_url
        self.spacy_model_name = spacy_model

        # Thread-local storage for models to ensure thread safety
        self._thread_local = threading.local()
        self._redis_client: redis.Redis | None = None

        if redis_url:
            try:
                self._redis_client = redis.from_url(redis_url)
                self._redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self._redis_client = None

    @property
    def sentence_transformer(self) -> SentenceTransformer:
        """Get thread-local SentenceTransformer instance."""
        if not hasattr(self._thread_local, "transformer"):
            try:
                self._thread_local.transformer = SentenceTransformer(self.MODEL_NAME)
                logger.info(f"Loaded SentenceTransformer model: {self.MODEL_NAME}")
            except Exception as e:
                raise ModelLoadingError(f"Failed to load transformer model: {e}")
        return self._thread_local.transformer

    @property
    def spacy_model(self) -> spacy.Language:
        """Get thread-local SpaCy model instance."""
        if not hasattr(self._thread_local, "spacy_nlp"):
            try:
                self._thread_local.spacy_nlp = spacy.load(self.spacy_model_name)
                logger.info(f"Loaded SpaCy model: {self.spacy_model_name}")
            except Exception as e:
                raise ModelLoadingError(f"Failed to load SpaCy model: {e}")
        return self._thread_local.spacy_nlp

    def _get_db_connection(self) -> psycopg2.extensions.connection:
        """Get database connection with exponential backoff retry."""
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                conn = psycopg2.connect(self.database_url)
                conn.autocommit = True
                return conn
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"DB connection attempt {attempt + 1} failed, retrying in {delay}s: {e}"
                )
                time.sleep(delay)

    def _clean_and_normalize_text(self, text: str) -> str:
        """Clean and normalize text for embedding generation.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""

        # Remove special characters but keep umlauts and essential punctuation
        text = re.sub(r"[^\w\s\.\,\-\:äöüÄÖÜß€]", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Convert to lowercase
        text = text.lower().strip()

        # Truncate to maximum length
        if len(text) > 8192:
            text = text[:8192]
            logger.warning("Text truncated to 8192 characters for embedding")

        return text

    def _extract_entities_with_spacy(self, text: str) -> Dict[str, List[str]]:
        """Extract key entities using SpaCy NER.

        Args:
            text: Text to process

        Returns:
            Dictionary of extracted entities by type
        """
        doc = self.spacy_model(text)
        entities = {
            "suppliers": [],
            "amounts": [],
            "dates": [],
            "locations": [],
            "products": [],
        }

        for ent in doc.ents:
            entity_text = ent.text.strip()
            if not entity_text:
                continue

            if ent.label_ in ["ORG", "PERSON"]:
                entities["suppliers"].append(entity_text)
            elif ent.label_ == "MONEY":
                entities["amounts"].append(entity_text)
            elif ent.label_ == "DATE":
                entities["dates"].append(entity_text)
            elif ent.label_ in ["GPE", "LOC"]:
                entities["locations"].append(entity_text)
            elif ent.label_ in ["PRODUCT", "MISC"]:
                entities["products"].append(entity_text)

        # Remove duplicates while preserving order
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))

        return entities

    def _create_structured_text(self, invoice_data: Dict[str, Any]) -> str:
        """Create structured text representation for embedding.

        Args:
            invoice_data: Invoice data dictionary

        Returns:
            Structured text optimized for embedding
        """
        text_parts = []

        # Add supplier information
        if "supplier_name" in invoice_data:
            text_parts.append(f"Lieferant: {invoice_data['supplier_name']}")

        # Add invoice details
        if "invoice_number" in invoice_data:
            text_parts.append(f"Rechnungsnummer: {invoice_data['invoice_number']}")

        if "amount" in invoice_data:
            text_parts.append(f"Betrag: {invoice_data['amount']}")

        if "date" in invoice_data:
            text_parts.append(f"Datum: {invoice_data['date']}")

        # Add description or items
        if "description" in invoice_data:
            text_parts.append(f"Beschreibung: {invoice_data['description']}")

        if "items" in invoice_data and isinstance(invoice_data["items"], list):
            items_text = " ".join([str(item) for item in invoice_data["items"]])
            text_parts.append(f"Positionen: {items_text}")

        return " | ".join(text_parts)

    def create_embedding(self, text: str, metadata: Dict[str, Any]) -> EmbeddingResult:
        """Create embedding for given text with metadata.

        Args:
            text: Text to create embedding for
            metadata: Additional metadata to store

        Returns:
            EmbeddingResult with embedding details

        Raises:
            DimensionError: If vector dimensions don't match expected values
            ModelLoadingError: If model loading fails
        """
        try:
            # Clean and normalize text
            cleaned_text = self._clean_and_normalize_text(text)
            if not cleaned_text:
                raise ValueError("Text is empty after cleaning")

            # Extract entities
            entities = self._extract_entities_with_spacy(cleaned_text)

            # Generate embedding
            start_time = time.time()
            embedding = self.sentence_transformer.encode(cleaned_text)
            embedding_time = time.time() - start_time

            # Validate dimensions
            if len(embedding) != self.VECTOR_DIMENSION:
                raise DimensionError(
                    f"Expected {self.VECTOR_DIMENSION} dimensions, got {len(embedding)}"
                )

            # Create result
            embedding_id = str(uuid.uuid4())
            result: EmbeddingResult = {
                "embedding_id": embedding_id,
                "vector": embedding.tolist(),
                "metadata": {
                    **metadata,
                    "entities": entities,
                    "text_length": len(cleaned_text),
                    "embedding_time": embedding_time,
                },
                "model_used": self.MODEL_NAME,
                "created_at": datetime.now(),
            }

            logger.info(f"Created embedding {embedding_id} in {embedding_time:.3f}s")
            return result

        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            raise

    def store_invoice_embedding(
        self, invoice_id: str, invoice_data: Dict[str, Any]
    ) -> StorageResult:
        """Store embedding for invoice data in database.

        Args:
            invoice_id: Unique invoice identifier
            invoice_data: Invoice data to create embedding from

        Returns:
            StorageResult with operation status
        """
        try:
            # Create structured text for embedding
            structured_text = self._create_structured_text(invoice_data)

            # Create embedding
            embedding_result = self.create_embedding(
                structured_text, {"invoice_id": invoice_id, **invoice_data}
            )

            # Store in database
            conn = self._get_db_connection()
            with conn.cursor() as cur:
                # Convert vector to pgvector format
                vector_str = f"[{','.join(map(str, embedding_result['vector']))}]"

                # Insert or update embedding
                query = """
                    INSERT INTO invoice_embeddings (id, invoice_id, embedding, metadata, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (invoice_id)
                    DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at
                """

                cur.execute(
                    query,
                    (
                        embedding_result["embedding_id"],
                        invoice_id,
                        vector_str,
                        embedding_result["metadata"],
                        embedding_result["created_at"],
                        embedding_result["created_at"],
                    ),
                )

            conn.close()

            # Invalidate cache
            if self._redis_client:
                try:
                    cache_pattern = f"embedding:*:{invoice_id}"
                    for key in self._redis_client.scan_iter(match=cache_pattern):
                        self._redis_client.delete(key)
                except Exception as e:
                    logger.warning(f"Cache invalidation failed: {e}")

            return StorageResult(
                success=True,
                embedding_id=embedding_result["embedding_id"],
                message="Embedding stored successfully",
            )

        except Exception as e:
            logger.error(f"Failed to store embedding for invoice {invoice_id}: {e}")
            return StorageResult(
                success=False, embedding_id="", message=f"Storage failed: {str(e)}"
            )

    def search_similar(
        self, query: str, limit: int = 10, threshold: float = 0.7
    ) -> List[SimilarityResult]:
        """Search for similar invoices using vector similarity.

        Args:
            query: Search query text
            limit: Maximum number of results
            threshold: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of SimilarityResult ordered by similarity
        """
        try:
            # Check cache first
            cache_key = f"search:{hash(query)}:{limit}:{threshold}"
            if self._redis_client:
                try:
                    cached_result = self._redis_client.get(cache_key)
                    if cached_result:
                        import json

                        return json.loads(cached_result)
                except Exception as e:
                    logger.warning(f"Cache retrieval failed: {e}")

            # Create query embedding
            cleaned_query = self._clean_and_normalize_text(query)
            query_embedding = self.sentence_transformer.encode(cleaned_query)

            if len(query_embedding) != self.VECTOR_DIMENSION:
                raise DimensionError("Query embedding dimension mismatch")

            # Search database
            conn = self._get_db_connection()
            with conn.cursor() as cur:
                # Use cosine similarity search with pgvector
                vector_str = f"[{','.join(map(str, query_embedding))}]"

                query_sql = """
                    SELECT invoice_id, metadata, (1 - (embedding <=> %s::vector)) as similarity_score
                    FROM invoice_embeddings
                    WHERE (1 - (embedding <=> %s::vector)) >= %s
                    ORDER BY similarity_score DESC
                    LIMIT %s
                """

                cur.execute(query_sql, (vector_str, vector_str, threshold, limit))
                results = cur.fetchall()

            conn.close()

            # Process results
            similarity_results = []
            for row in results:
                invoice_id, metadata, similarity_score = row

                # Extract matched fields based on entities
                query_entities = self._extract_entities_with_spacy(cleaned_query)
                stored_entities = metadata.get("entities", {})

                matched_fields = []
                for field_type in query_entities:
                    if (
                        query_entities[field_type]
                        and stored_entities.get(field_type)
                        and any(
                            entity in stored_entities[field_type]
                            for entity in query_entities[field_type]
                        )
                    ):
                        matched_fields.append(field_type)

                similarity_results.append(
                    SimilarityResult(
                        invoice_id=invoice_id,
                        similarity_score=float(similarity_score),
                        invoice_data=metadata,
                        matched_fields=matched_fields,
                    )
                )

            # Cache results
            if self._redis_client and similarity_results:
                try:
                    import json

                    # Convert datetime objects to strings for JSON serialization
                    cacheable_results = []
                    for result in similarity_results:
                        cacheable_result = result.copy()
                        # Handle datetime serialization in metadata
                        if "created_at" in cacheable_result["invoice_data"]:
                            created_at = cacheable_result["invoice_data"]["created_at"]
                            if isinstance(created_at, datetime):
                                cacheable_result["invoice_data"]["created_at"] = (
                                    created_at.isoformat()
                                )
                        cacheable_results.append(cacheable_result)

                    self._redis_client.setex(
                        cache_key,
                        300,  # 5 minutes
                        json.dumps(cacheable_results),
                    )
                except Exception as e:
                    logger.warning(f"Cache storage failed: {e}")

            logger.info(f"Found {len(similarity_results)} similar invoices for query")
            return similarity_results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def update_embedding(
        self, invoice_id: str, new_data: Dict[str, Any]
    ) -> UpdateResult:
        """Update existing embedding with new invoice data.

        Args:
            invoice_id: Invoice ID to update
            new_data: New invoice data

        Returns:
            UpdateResult with operation status
        """
        try:
            # Create new embedding
            structured_text = self._create_structured_text(new_data)
            embedding_result = self.create_embedding(
                structured_text, {"invoice_id": invoice_id, **new_data}
            )

            # Update in database
            conn = self._get_db_connection()
            with conn.cursor() as cur:
                vector_str = f"[{','.join(map(str, embedding_result['vector']))}]"

                query = """
                    UPDATE invoice_embeddings
                    SET embedding = %s, metadata = %s, updated_at = %s
                    WHERE invoice_id = %s
                    RETURNING id
                """

                cur.execute(
                    query,
                    (
                        vector_str,
                        embedding_result["metadata"],
                        datetime.now(),
                        invoice_id,
                    ),
                )

                if not cur.fetchone():
                    raise ValueError(f"Invoice {invoice_id} not found")

            conn.close()

            # Invalidate cache
            if self._redis_client:
                try:
                    cache_patterns = [f"embedding:*:{invoice_id}", "search:*"]
                    for pattern in cache_patterns:
                        for key in self._redis_client.scan_iter(match=pattern):
                            self._redis_client.delete(key)
                except Exception as e:
                    logger.warning(f"Cache invalidation failed: {e}")

            return UpdateResult(
                success=True,
                embedding_id=embedding_result["embedding_id"],
                updated_at=datetime.now(),
                message="Embedding updated successfully",
            )

        except Exception as e:
            logger.error(f"Failed to update embedding for invoice {invoice_id}: {e}")
            return UpdateResult(
                success=False,
                embedding_id="",
                updated_at=datetime.now(),
                message=f"Update failed: {str(e)}",
            )

    def batch_create_embeddings(
        self, items: List[Tuple[str, Dict]]
    ) -> List[EmbeddingResult]:
        """Create embeddings for multiple items in batch.

        Args:
            items: List of (text, metadata) tuples

        Returns:
            List of EmbeddingResult objects
        """
        results = []
        batch_size = 100  # Process in chunks to avoid OOM

        try:
            for i in range(0, len(items), batch_size):
                batch = items[i : i + batch_size]
                batch_results = []

                # Prepare texts for batch encoding
                texts = []
                metadatas = []

                for text, metadata in batch:
                    cleaned_text = self._clean_and_normalize_text(text)
                    if cleaned_text:
                        texts.append(cleaned_text)
                        metadatas.append(metadata)

                if not texts:
                    continue

                # Batch encode embeddings
                start_time = time.time()
                embeddings = self.sentence_transformer.encode(texts)
                batch_time = time.time() - start_time

                # Process individual results
                for j, (text, metadata) in enumerate(
                    zip(texts, metadatas, strict=False)
                ):
                    embedding = embeddings[j]

                    if len(embedding) != self.VECTOR_DIMENSION:
                        logger.warning(f"Skipping item {i+j} due to dimension mismatch")
                        continue

                    # Extract entities
                    entities = self._extract_entities_with_spacy(text)

                    embedding_id = str(uuid.uuid4())
                    result: EmbeddingResult = {
                        "embedding_id": embedding_id,
                        "vector": embedding.tolist(),
                        "metadata": {
                            **metadata,
                            "entities": entities,
                            "text_length": len(text),
                            "batch_time": batch_time / len(texts),
                        },
                        "model_used": self.MODEL_NAME,
                        "created_at": datetime.now(),
                    }

                    batch_results.append(result)

                results.extend(batch_results)
                logger.info(
                    f"Processed batch {i//batch_size + 1}: {len(batch_results)} embeddings in {batch_time:.3f}s"
                )

            logger.info(f"Batch processing complete: {len(results)} embeddings created")
            return results

        except Exception as e:
            logger.error(f"Batch embedding creation failed: {e}")
            raise

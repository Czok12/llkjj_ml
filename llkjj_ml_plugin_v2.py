"""
LLKJJ ML Plugin v2.0 - Stateless Repository-Pattern Implementation

Refactored Version des ML-Plugins für Integration mit llkjj_backend Repository-Pattern.

**Breaking Changes von v1.x:**
- Stateless Design (keine interne Datenpersistierung)
- Repository-Pattern via Dependency Injection
- Elimination der ResourceManager-Singleton
- Gemini-Direct ohne Docling als Standard

Author: LLKJJ ML Team
Version: 2.0.0
Date: 2025-01-25
"""

import logging
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable
from uuid import UUID
import asyncio
from pydantic import BaseModel

logger = logging.getLogger(__name__)

__version__ = "2.0.0"
__all__ = ["MLPlugin", "MLPluginConfig", "InvoiceProcessingResult", "GeminiClient", "EmbeddingProvider", "TrainingFeedback", "TrainingItem", "EmbeddingData", "SimilarItem"]


# ============================================================================= 
# Protocol Definitions für Dependency Injection
# =============================================================================

@runtime_checkable
class MLRepositoryInterface(Protocol):
    """Protocol für ML-Repository aus llkjj_backend."""
    
    async def store_invoice_embedding(
        self, 
        invoice_id: UUID, 
        invoice_item_id: UUID,
        embedding_data: "EmbeddingData"
    ) -> UUID:
        """Speichere Embedding für Rechungsposition in pgvector."""
        ...
    
    async def query_similar_items(
        self, 
        query_text: str, 
        query_embedding: list[float], 
        limit: int = 5,
        min_similarity: float = 0.7
    ) -> list["SimilarItem"]:
        """Finde ähnliche Rechnungspositionen via pgvector cosine similarity."""
        ...
    
    async def get_embedding_stats(self) -> dict:
        """Statistiken über gespeicherte Embeddings."""
        ...
    
    async def store_training_feedback(self, feedback: "TrainingFeedback") -> UUID:
        """Speichere User-Feedback für spaCy-Training."""
        ...
    
    async def get_pending_training_data(
        self, 
        limit: Optional[int] = None,
        min_confidence_rating: int = 3
    ) -> list["TrainingItem"]:
        """Hole unbenutzte Feedback-Daten für spaCy-Training."""
        ...
    
    async def mark_training_data_used(self, feedback_ids: list[UUID]) -> None:
        """Markiere Feedback-Daten als 'used_in_training'."""
        ...
    
    async def get_training_stats(self) -> dict:
        """Statistiken über Training-Feedback."""
        ...
    
    async def store_spacy_model(self, model_name: str, model_path: Path, version: str, metrics: dict) -> UUID:
        """Speichere trainiertes spaCy-Modell."""
        ...
    
    async def get_latest_spacy_model(self, model_type: str) -> Optional[dict]:
        """Hole neuestes spaCy-Modell für gegebenen Typ."""
        ...
    
    async def log_training_run(self, config: dict, metrics: dict, model_id: UUID) -> UUID:
        """Protokolliere Training-Lauf."""
        ...


@runtime_checkable
class GeminiClient(Protocol):
    """Protocol für Gemini-Client Abhängigkeit."""
    
    async def process_pdf_direct(self, pdf_path: Path) -> dict:
        """Verarbeite PDF direkt mit Gemini (ohne Docling)."""
        ...
    
    async def classify_with_rag_context(self, item_description: str, context: dict) -> dict:
        """Klassifiziere Item mit RAG-Kontext."""
        ...


@runtime_checkable 
class EmbeddingProvider(Protocol):
    """Protocol für Embedding-Provider."""
    
    def encode(self, text: str) -> list[float]:
        """Encode Text zu Embedding-Vector."""
        ...


# ============================================================================= 
# Pydantic Data Models (import from llkjj_backend)
# =============================================================================

class EmbeddingData(BaseModel):
    """Embedding-Daten für pgvector-Speicherung."""
    content_text: str
    embedding_vector: list[float]  # 384-dimensional
    supplier_name: str
    item_description: str
    article_number: Optional[str] = None
    skr03_account: str
    skr03_category: str
    classification_confidence: float


class SimilarItem(BaseModel):
    """Ähnliches Item aus RAG-System."""
    invoice_id: UUID
    item_description: str
    supplier_name: str
    skr03_account: str
    skr03_category: str
    similarity_score: float  # Cosine similarity
    classification_confidence: float


class TrainingFeedback(BaseModel):
    """User-Feedback für spaCy-Training."""
    invoice_item_id: UUID
    user_id: UUID
    original_classification: str
    corrected_classification: str
    feedback_type: str  # 'manual_correction', 'approval', 'rejection'
    user_confidence_rating: int  # 1-5
    item_description: str
    supplier_context: str
    amount_context: Optional[float] = None


class TrainingItem(BaseModel):
    """Training-Item aus Repository."""
    id: UUID
    item_description: str
    supplier_context: str
    correct_classification: str
    confidence_score: float
    created_at: str  # ISO timestamp


# ============================================================================= 
# Plugin Configuration
# =============================================================================

class MLPluginConfig(BaseModel):
    """Konfiguration für ML-Plugin."""
    validate_environment: bool = True
    enable_rag_enhancement: bool = True
    similarity_threshold: float = 0.75
    max_rag_context_items: int = 5
    enable_gemini_direct: bool = True  # True = ohne Docling, False = mit Docling
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.1
    max_output_tokens: int = 8192


class InvoiceProcessingResult(BaseModel):
    """Ergebnis der Rechnungsverarbeitung."""
    invoice_id: UUID
    processing_successful: bool
    items_processed: int
    items_with_rag_enhancement: int
    average_confidence: float
    processing_time_ms: int
    errors: list[str] = []
    metadata: dict = {}


# ============================================================================= 
# Stateless ML Plugin Implementation
# =============================================================================

class MLPlugin:
    """
    LLKJJ ML Plugin v2.0 - Stateless Repository-Pattern Implementation
    
    **Stateless Design:**
    - Keine eigene Datenpersistierung
    - Repository-Pattern für DB-Zugriff
    - Dependency Injection für alle Services
    - Explizite Resource-Verwaltung (keine Singletons)
    
    **Kernfunktionen:**
    - PDF → Gemini Direct (ohne Docling als Standard)
    - RAG-Enhancement via pgvector
    - Training-Feedback für spaCy
    
    **Breaking Changes von v1.x:**
    - Requires MLRepositoryInterface injection
    - Eliminiert ResourceManager-Singleton
    - Async-only processing methods
    - Neue Dependency-Injection-Pattern
    """
    
    def __init__(
        self,
        repository: MLRepositoryInterface,
        gemini_client: GeminiClient,
        embedding_provider: EmbeddingProvider,
        config: Optional[MLPluginConfig] = None
    ):
        """
        Initialisierung mit Dependency Injection.
        
        Args:
            repository: ML-Repository für DB-Operationen (aus llkjj_backend)
            gemini_client: Gemini API Client
            embedding_provider: Sentence-Transformers Provider
            config: Plugin-Konfiguration
        """
        self.repository = repository
        self.gemini_client = gemini_client
        self.embedding_provider = embedding_provider
        self.config = config or MLPluginConfig()
        
        # Validate Protocol compliance
        if not isinstance(repository, MLRepositoryInterface):
            raise TypeError(f"repository must implement MLRepositoryInterface, got {type(repository)}")
        if not isinstance(gemini_client, GeminiClient):
            raise TypeError(f"gemini_client must implement GeminiClient, got {type(gemini_client)}")  
        if not isinstance(embedding_provider, EmbeddingProvider):
            raise TypeError(f"embedding_provider must implement EmbeddingProvider, got {type(embedding_provider)}")
        
        logger.info("✅ MLPlugin v2.0 initialisiert (Stateless, Repository-Pattern)")
    
    async def process_invoice_pdf(self, pdf_path: Path, invoice_id: UUID) -> InvoiceProcessingResult:
        """
        Hauptverarbeitungslogik: PDF → Gemini → SKR03 → pgvector
        
        Args:
            pdf_path: Pfad zur PDF-Datei
            invoice_id: UUID der Rechnung im llkjj_backend
        
        Returns:
            InvoiceProcessingResult mit Statistiken
        """
        import time
        start_time = time.time()
        
        result = InvoiceProcessingResult(
            invoice_id=invoice_id,
            processing_successful=False,
            items_processed=0,
            items_with_rag_enhancement=0,
            average_confidence=0.0,
            processing_time_ms=0
        )
        
        try:
            # Phase 1: Gemini Direct Processing (ohne Docling)
            logger.info(f"🚀 Starte Gemini-Direct-Processing: {pdf_path}")
            gemini_result = await self.gemini_client.process_pdf_direct(pdf_path)
            
            if not gemini_result.get('success', False):
                result.errors.append("Gemini PDF-Processing fehlgeschlagen")
                return result
            
            invoice_items = gemini_result.get('invoice_items', [])
            if not invoice_items:
                result.errors.append("Keine Rechnungspositionen extrahiert")
                return result
            
            # Phase 2: RAG-Enhancement für jede Position
            confidences = []
            rag_enhanced_count = 0
            
            for item_index, item in enumerate(invoice_items):
                try:
                    # RAG-Enhancement wenn aktiviert
                    if self.config.enable_rag_enhancement:
                        enhanced_classification = await self._enhance_with_rag(item)
                        if enhanced_classification:
                            item.update(enhanced_classification)
                            rag_enhanced_count += 1
                    
                    # Embedding speichern für zukünftige RAG-Queries
                    await self._store_item_embedding(invoice_id, item_index, item)
                    
                    confidences.append(item.get('classification_confidence', 0.0))
                    
                except Exception as e:
                    logger.error(f"Fehler bei Item {item_index}: {e}")
                    result.errors.append(f"Item {item_index}: {str(e)}")
            
            # Ergebnis zusammenstellen
            result.processing_successful = len(result.errors) == 0
            result.items_processed = len(invoice_items)
            result.items_with_rag_enhancement = rag_enhanced_count
            result.average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            result.processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Metadata hinzufügen
            result.metadata = {
                "architecture": "stateless_repository",
                "ml_module": f"llkjj_ml v{__version__}",
                "gemini_model": self.config.gemini_model,
                "rag_enabled": self.config.enable_rag_enhancement,
                "docling_enabled": not self.config.enable_gemini_direct
            }
            
            logger.info(f"✅ Verarbeitung erfolgreich: {result.items_processed} Items, "
                       f"{result.items_with_rag_enhancement} RAG-enhanced, "
                       f"Ø Confidence: {result.average_confidence:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Kritischer Fehler bei PDF-Verarbeitung: {e}")
            result.errors.append(f"Kritischer Fehler: {str(e)}")
            result.processing_time_ms = int((time.time() - start_time) * 1000)
            return result
    
    async def _enhance_with_rag(self, invoice_item: dict) -> Optional[dict]:
        """
        Verbessere SKR03-Klassifizierung mit RAG-System.
        
        Args:
            invoice_item: Rechnungsposition von Gemini
        
        Returns:
            Enhanced classification oder None
        """
        try:
            item_description = invoice_item.get('description', '')
            if not item_description:
                return None
            
            # Embedding für Suche erstellen
            query_embedding = self.embedding_provider.encode(item_description)
            
            # Ähnliche Items via pgvector suchen
            similar_items = await self.repository.query_similar_items(
                query_text=item_description,
                query_embedding=query_embedding,
                limit=self.config.max_rag_context_items,
                min_similarity=self.config.similarity_threshold
            )
            
            if not similar_items:
                logger.debug(f"Keine ähnlichen Items für: {item_description[:50]}...")
                return None
            
            # RAG-Kontext für Gemini aufbauen
            rag_context = {
                'similar_items': [
                    {
                        'description': item.item_description,
                        'supplier': item.supplier_name,
                        'skr03_account': item.skr03_account,
                        'confidence': item.classification_confidence,
                        'similarity': item.similarity_score
                    }
                    for item in similar_items
                ],
                'context_strength': len(similar_items),
                'average_similarity': sum(item.similarity_score for item in similar_items) / len(similar_items)
            }
            
            # Gemini mit RAG-Kontext aufrufen
            enhanced_result = await self.gemini_client.classify_with_rag_context(
                item_description, 
                rag_context
            )
            
            if enhanced_result.get('success', False):
                logger.debug(f"✅ RAG-Enhancement für: {item_description[:50]}... "
                           f"({len(similar_items)} ähnliche Items)")
                return {
                    'skr03_account': enhanced_result.get('skr03_account'),
                    'classification_confidence': enhanced_result.get('confidence'),
                    'rag_enhanced': True,
                    'rag_context_items': len(similar_items)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Fehler bei RAG-Enhancement: {e}")
            return None
    
    async def _store_item_embedding(
        self, 
        invoice_id: UUID, 
        item_index: int, 
        invoice_item: dict
    ) -> None:
        """Speichere Embedding für Rechnungsposition."""
        try:
            # Embedding erstellen
            content_text = f"Lieferant: {invoice_item.get('supplier', '')} | " \
                          f"Artikel: {invoice_item.get('description', '')}"
            embedding_vector = self.embedding_provider.encode(content_text)
            
            # EmbeddingData erstellen
            embedding_data = EmbeddingData(
                content_text=content_text,
                embedding_vector=embedding_vector,
                supplier_name=invoice_item.get('supplier', ''),
                item_description=invoice_item.get('description', ''),
                article_number=invoice_item.get('article_number'),
                skr03_account=invoice_item.get('skr03_account', ''),
                skr03_category=invoice_item.get('skr03_category', ''),
                classification_confidence=invoice_item.get('classification_confidence', 0.0)
            )
            
            # In Repository speichern (pgvector)
            await self.repository.store_invoice_embedding(
                invoice_id=invoice_id,
                invoice_item_id=UUID(f"00000000-0000-0000-0000-{item_index:012d}"),  # Temporary ID
                embedding_data=embedding_data
            )
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Embeddings: {e}")
    
    async def submit_training_feedback(
        self, 
        invoice_item_id: UUID,
        user_id: UUID,
        original_classification: str,
        corrected_classification: str,
        confidence_rating: int,
        feedback_context: dict
    ) -> UUID:
        """
        Speichere User-Feedback für spaCy-Training.
        
        Args:
            invoice_item_id: ID der korrigierten Rechnungsposition
            user_id: ID des Users
            original_classification: Original SKR03-Konto
            corrected_classification: Korrigiertes SKR03-Konto
            confidence_rating: User-Confidence (1-5)
            feedback_context: Zusätzlicher Kontext
        
        Returns:
            UUID des gespeicherten Feedbacks
        """
        feedback = TrainingFeedback(
            invoice_item_id=invoice_item_id,
            user_id=user_id,
            original_classification=original_classification,
            corrected_classification=corrected_classification,
            feedback_type='manual_correction' if original_classification != corrected_classification else 'approval',
            user_confidence_rating=confidence_rating,
            item_description=feedback_context.get('item_description', ''),
            supplier_context=feedback_context.get('supplier_name', ''),
            amount_context=feedback_context.get('amount')
        )
        
        feedback_id = await self.repository.store_training_feedback(feedback)
        
        logger.info(f"✅ Training-Feedback gespeichert: {original_classification} → {corrected_classification}")
        return feedback_id
    
    async def get_embedding_statistics(self) -> dict:
        """Hole Embedding-Statistiken aus Repository."""
        return await self.repository.get_embedding_stats()
    
    async def get_training_statistics(self) -> dict:
        """Hole Training-Statistiken aus Repository."""
        return await self.repository.get_training_stats()
    
    async def start_spacy_training(
        self, 
        model_name: str = "skr03_classifier",
        min_confidence_rating: int = 3,
        max_training_items: Optional[int] = None
    ) -> dict:
        """
        Starte spaCy-Training mit verfügbaren Feedback-Daten.
        
        Args:
            model_name: Name des zu trainierenden Modells
            min_confidence_rating: Minimale User-Confidence (1-5)
            max_training_items: Maximale Anzahl Training-Items
        
        Returns:
            Training-Ergebnis mit Metriken
        """
        from .src.training.spacy_trainer import SpacyTrainer, TrainingConfig
        
        try:
            logger.info(f"🏃 Starte spaCy-Training für {model_name}...")
            
            # Training-Konfiguration erstellen
            training_config = TrainingConfig(
                model_name=model_name,
                min_confidence_rating=min_confidence_rating,
                max_training_items=max_training_items or 1000,
                validation_split=0.2,
                batch_size=32,
                max_epochs=50,
                learning_rate=0.001,
                dropout=0.1,
                early_stopping_patience=5,
                elektrohandwerk_optimization=True
            )
            
            # SpacyTrainer initialisieren
            trainer = SpacyTrainer(
                repository=self.repository,
                config=training_config
            )
            
            # Training durchführen
            training_metrics = await trainer.train_classification_model(model_name)
            
            logger.info(f"✅ spaCy-Training abgeschlossen: "
                       f"Accuracy={training_metrics.final_accuracy:.3f}, "
                       f"F1={training_metrics.final_f1_score:.3f}")
            
            return {
                "success": True,
                "model_name": model_name,
                "training_items_used": training_metrics.training_items_used,
                "final_accuracy": training_metrics.final_accuracy,
                "final_f1_score": training_metrics.final_f1_score,
                "training_duration_minutes": training_metrics.training_duration_minutes,
                "model_path": str(training_metrics.model_save_path) if training_metrics.model_save_path else None,
                "epochs_completed": training_metrics.epochs_completed
            }
            
        except Exception as e:
            logger.error(f"❌ spaCy-Training fehlgeschlagen: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_name": model_name
            }
    
    async def get_training_readiness(self) -> dict:
        """
        Prüfe Bereitschaft für spaCy-Training.
        
        Returns:
            Training-Readiness mit Statistiken
        """
        try:
            # Verfügbare Training-Daten abrufen (nur zählen)
            training_data = await self.repository.get_pending_training_data(limit=1, min_confidence_rating=1)
            has_training_data = len(training_data) > 0
            
            # Training-Statistiken
            training_stats = await self.repository.get_training_stats()
            
            # Embedding-Statistiken
            embedding_stats = await self.repository.get_embedding_stats()
            
            readiness_score = 0
            readiness_checks = {
                "has_pending_feedback": has_training_data,
                "sufficient_high_confidence_feedback": training_stats.get("high_confidence_count", 0) >= 50,
                "diverse_supplier_feedback": training_stats.get("unique_suppliers", 0) >= 10,
                "recent_feedback_activity": training_stats.get("recent_feedback_count", 0) >= 20,
                "embedding_diversity": embedding_stats.get("unique_skr03_accounts", 0) >= 15
            }
            
            readiness_score = sum(readiness_checks.values()) / len(readiness_checks)
            
            return {
                "ready_for_training": readiness_score >= 0.6,  # 60% der Checks erfüllt
                "readiness_score": readiness_score,
                "readiness_checks": readiness_checks,
                "training_stats": training_stats,
                "embedding_stats": embedding_stats,
                "recommendation": (
                    "Training empfohlen" if readiness_score >= 0.8 else
                    "Training möglich" if readiness_score >= 0.6 else
                    "Mehr Training-Daten sammeln"
                )
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Training-Readiness-Check: {e}")
            return {
                "ready_for_training": False,
                "error": str(e)
            }
    
    def get_plugin_info(self) -> dict:
        """Plugin-Informationen für Registry."""
        return {
            'name': 'LLKJJ ML Plugin',
            'version': __version__,
            'type': 'stateless_repository',
            'capabilities': [
                'pdf_processing',
                'gemini_direct',
                'rag_enhancement',
                'training_feedback',
                'pgvector_embeddings',
                'spacy_training',
                'training_readiness'
            ],
            'dependencies': [
                'MLRepositoryInterface',
                'GeminiClient', 
                'EmbeddingProvider'
            ]
        }


# ============================================================================= 
# Backward Compatibility Bridge (Optional)
# =============================================================================

def create_plugin_from_backend_registry(
    backend_registry, 
    config: Optional[MLPluginConfig] = None
) -> MLPlugin:
    """
    Convenience Funktion zur Plugin-Erstellung über llkjj_backend Registry.
    
    Args:
        backend_registry: Module-Registry aus llkjj_backend
        config: Optional Plugin-Konfiguration
    
    Returns:
        Configured MLPlugin instance
    
    Example:
        >>> from llkjj_backend.core.modules import registry
        >>> plugin = create_plugin_from_backend_registry(registry)
        >>> result = await plugin.process_invoice_pdf(pdf_path, invoice_id)
    """
    # Repository aus Registry abrufen
    repository = backend_registry.get_service('ml_repository')
    if not repository:
        raise RuntimeError("ML Repository nicht in Backend-Registry verfügbar")
    
    # Weitere Services aus Registry (falls verfügbar)
    gemini_client = backend_registry.get_service('gemini') 
    embedding_provider = backend_registry.get_service('embeddings')
    
    if not gemini_client or not embedding_provider:
        raise RuntimeError("Gemini Client oder Embedding Provider nicht in Registry verfügbar")
    
    return MLPlugin(
        repository=repository,
        gemini_client=gemini_client,
        embedding_provider=embedding_provider,
        config=config
    )
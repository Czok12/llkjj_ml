# LLKJJ Machine Learning

**Version:** 0.1.0
**Beschreibung:** KI-gestützte Machine Learning Services für die LLKJJ Finanzbuchhaltung

## Überblick

Das `llkjj_ml` Paket stellt fortschrittliche Machine Learning-Funktionen für die automatisierte Finanzbuchhaltung bereit. Es umfasst intelligente Dokumentenverarbeitung, automatische Klassifizierung, Buchungsvorschläge und prädiktive Analysen.

## Architektur

### ML Pipeline Architecture
- **Document Processing:** Intelligente Dokumentenanalyse
- **Feature Extraction:** Automatische Merkmalserkennung
- **Classification Models:** Mehrschichtige Klassifizierung
- **Prediction Engine:** Vorhersage-Algorithmen
- **Continuous Learning:** Adaptives Modell-Training

### Verzeichnisstruktur

```
llkjj_ml/
├── src/llkjj_ml/
│   ├── core/
│   │   ├── base_processor.py      # Basis-Verarbeitungsklassen
│   │   ├── model_registry.py     # Modell-Verwaltung
│   │   └── pipeline_manager.py   # Pipeline-Orchestrierung
│   ├── extraction/               # Datenextraktion
│   │   ├── docling_processor.py  # Docling-Integration
│   │   ├── text_extractor.py     # Text-Extraktion
│   │   └── entity_extractor.py   # Entity-Recognition
│   ├── classification/           # Klassifizierung
│   │   ├── document_classifier.py # Dokumenten-Klassifizierer
│   │   ├── invoice_classifier.py  # Rechnungs-Klassifizierer
│   │   └── rule_engine.py         # Regel-basierte Klassifizierung
│   ├── embedding/               # Vector Embeddings
│   │   ├── sentence_transformer.py # Sentence Transformers
│   │   ├── embedding_service.py    # Embedding-Service
│   │   └── vector_store.py         # Vector Database
│   ├── prediction/              # Vorhersage-Models
│   │   ├── booking_predictor.py   # Buchungsvorhersage
│   │   ├── amount_predictor.py    # Betragsprognose
│   │   └── trend_analyzer.py      # Trend-Analyse
│   ├── training/                # Model Training
│   │   ├── trainer.py            # Modell-Trainer
│   │   ├── data_preprocessor.py  # Datenvorverarbeitung
│   │   └── evaluation.py         # Modell-Evaluation
│   ├── optimization/            # Performance-Optimierung
│   │   ├── model_optimizer.py    # Modell-Optimierung
│   │   ├── batch_processor.py    # Batch-Verarbeitung
│   │   └── cache_manager.py      # Cache-Management
│   └── __init__.py
├── models/                      # Trainierte Modelle
│   ├── classification/
│   ├── embedding/
│   └── prediction/
├── ml_service/                  # ML Service CLI
│   └── cli.py
└── pyproject.toml
```

## Core ML Services

### Document Processing

#### Intelligent Text Extraction
```python
class IntelligentTextExtractor:
    """Advanced text extraction with ML enhancement"""

    async def extract_structured_data(self, document: bytes) -> ExtractedData:
        # OCR + ML-enhanced text recognition
        raw_text = await self.ocr_engine.extract(document)
        structured_data = await self.ml_enhancer.structure(raw_text)

        return ExtractedData(
            text=structured_data.text,
            entities=structured_data.entities,
            confidence=structured_data.confidence
        )
```

#### Entity Recognition
```python
class FinancialEntityExtractor:
    """Extract financial entities from documents"""

    def extract_entities(self, text: str) -> List[Entity]:
        entities = []

        # Amount extraction
        amounts = self.extract_amounts(text)
        entities.extend(amounts)

        # Date extraction
        dates = self.extract_dates(text)
        entities.extend(dates)

        # Company extraction
        companies = self.extract_companies(text)
        entities.extend(companies)

        return entities
```

### Classification Services

#### Multi-Level Document Classification
```python
class HierarchicalDocumentClassifier:
    """Hierarchical classification system"""

    async def classify(self, document: Document) -> ClassificationResult:
        # Level 1: Document type (invoice, receipt, contract)
        doc_type = await self.type_classifier.predict(document)

        # Level 2: Subtype classification
        if doc_type == "invoice":
            subtype = await self.invoice_classifier.predict(document)
        elif doc_type == "receipt":
            subtype = await self.receipt_classifier.predict(document)

        # Level 3: Business category
        category = await self.category_classifier.predict(document, doc_type)

        return ClassificationResult(
            document_type=doc_type,
            subtype=subtype,
            category=category,
            confidence_scores={
                "type": doc_type.confidence,
                "subtype": subtype.confidence,
                "category": category.confidence
            }
        )
```

#### Rule-Based Classification Engine
```python
class FinancialRuleEngine:
    """Rule-based classification with ML fallback"""

    def __init__(self):
        self.rules = [
            InvoiceAmountRule(),
            TaxRateRule(),
            CompanyNameRule(),
            DateValidationRule()
        ]

    async def classify_with_rules(self, document: Document) -> RuleResult:
        results = []
        for rule in self.rules:
            result = await rule.evaluate(document)
            results.append(result)

        # Combine rule results with ML predictions
        final_class = self.combine_results(results)
        return final_class
```

### Embedding & Semantic Search

#### Sentence Transformer Integration
```python
class SemanticEmbeddingService:
    """Generate semantic embeddings for documents"""

    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    async def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        # Batch processing for efficiency
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    async def semantic_search(self, query: str, documents: List[Document]) -> List[SearchResult]:
        query_embedding = self.model.encode([query])
        doc_embeddings = [doc.embedding for doc in documents]

        # Cosine similarity search
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

        results = []
        for idx, similarity in enumerate(similarities):
            if similarity > 0.7:  # Threshold
                results.append(SearchResult(
                    document=documents[idx],
                    similarity=similarity
                ))

        return sorted(results, key=lambda x: x.similarity, reverse=True)
```

### Prediction Services

#### Intelligent Booking Predictor
```python
class BookingPredictor:
    """Predict optimal booking suggestions"""

    async def predict_booking(self, invoice_data: InvoiceData) -> BookingPrediction:
        # Feature engineering
        features = self.extract_features(invoice_data)

        # Account prediction
        predicted_account = await self.account_predictor.predict(features)

        # Cost center prediction
        predicted_cost_center = await self.cost_center_predictor.predict(features)

        # Tax rate prediction
        predicted_tax_rate = await self.tax_predictor.predict(features)

        return BookingPrediction(
            debit_account=predicted_account.debit,
            credit_account=predicted_account.credit,
            cost_center=predicted_cost_center,
            tax_rate=predicted_tax_rate,
            confidence=min(
                predicted_account.confidence,
                predicted_cost_center.confidence,
                predicted_tax_rate.confidence
            )
        )
```

#### Financial Trend Analysis
```python
class TrendAnalyzer:
    """Analyze financial trends and patterns"""

    async def analyze_spending_patterns(self, user_id: int, period: DateRange) -> TrendAnalysis:
        # Historical data retrieval
        transactions = await self.get_transactions(user_id, period)

        # Time series analysis
        monthly_spending = self.aggregate_by_month(transactions)
        trend = self.calculate_trend(monthly_spending)

        # Anomaly detection
        anomalies = self.detect_anomalies(monthly_spending)

        # Predictions
        future_spending = self.predict_future_spending(monthly_spending)

        return TrendAnalysis(
            trend_direction=trend.direction,
            trend_strength=trend.strength,
            anomalies=anomalies,
            predictions=future_spending,
            insights=self.generate_insights(trend, anomalies)
        )
```

## Model Training & Management

### Training Pipeline
```python
class MLTrainingPipeline:
    """Comprehensive model training pipeline"""

    async def train_classification_model(self, training_data: TrainingData) -> TrainedModel:
        # Data preprocessing
        preprocessed_data = await self.preprocessor.process(training_data)

        # Feature engineering
        features = await self.feature_engineer.extract(preprocessed_data)

        # Model training
        model = await self.trainer.train(features)

        # Model evaluation
        metrics = await self.evaluator.evaluate(model, features.test_set)

        # Model registration
        model_id = await self.model_registry.register(model, metrics)

        return TrainedModel(
            id=model_id,
            model=model,
            metrics=metrics,
            version=self.get_next_version()
        )
```

### Continuous Learning
```python
class ContinuousLearner:
    """Implement continuous learning from user feedback"""

    async def learn_from_correction(self,
                                   prediction: Prediction,
                                   correction: UserCorrection):
        # Store correction as training example
        training_example = TrainingExample(
            input_features=prediction.features,
            predicted_output=prediction.output,
            correct_output=correction.correct_output,
            user_id=correction.user_id,
            timestamp=datetime.utcnow()
        )

        await self.training_store.save(training_example)

        # Trigger retraining if threshold reached
        correction_count = await self.training_store.count_corrections()
        if correction_count >= self.retrain_threshold:
            await self.trigger_retraining()
```

## Performance Optimization

### Model Optimization
```python
class ModelOptimizer:
    """Optimize ML models for production deployment"""

    async def optimize_for_inference(self, model: MLModel) -> OptimizedModel:
        # Model quantization
        quantized_model = self.quantize_model(model)

        # Model pruning
        pruned_model = self.prune_model(quantized_model)

        # ONNX conversion for faster inference
        onnx_model = self.convert_to_onnx(pruned_model)

        return OptimizedModel(
            model=onnx_model,
            optimization_metrics=self.calculate_optimization_metrics(model, onnx_model)
        )
```

### Batch Processing
```python
class BatchMLProcessor:
    """Efficient batch processing for ML operations"""

    async def process_document_batch(self, documents: List[Document]) -> List[ProcessingResult]:
        # Batch OCR processing
        ocr_results = await self.batch_ocr(documents)

        # Batch classification
        classifications = await self.batch_classify(ocr_results)

        # Batch embedding generation
        embeddings = await self.batch_embed(ocr_results)

        # Combine results
        results = []
        for i, doc in enumerate(documents):
            results.append(ProcessingResult(
                document=doc,
                ocr_result=ocr_results[i],
                classification=classifications[i],
                embedding=embeddings[i]
            ))

        return results
```

## Integration & Deployment

### API Integration
```python
# ML Service API
@app.post("/ml/classify-document")
async def classify_document(file: UploadFile) -> ClassificationResponse:
    """Classify uploaded document using ML"""

    # Document processing
    processed_doc = await ml_service.process_document(file)

    # Classification
    classification = await ml_service.classify(processed_doc)

    # Entity extraction
    entities = await ml_service.extract_entities(processed_doc)

    return ClassificationResponse(
        classification=classification,
        entities=entities,
        confidence=classification.confidence
    )

@app.post("/ml/predict-booking")
async def predict_booking(invoice_data: InvoiceData) -> BookingPrediction:
    """Predict optimal booking for invoice"""

    prediction = await ml_service.predict_booking(invoice_data)
    return prediction
```

### Model Serving
```python
class MLModelServer:
    """Serve ML models for real-time inference"""

    def __init__(self):
        self.models = {}
        self.load_models()

    async def load_models(self):
        """Load all production models"""
        self.models['document_classifier'] = await self.load_model('document_classifier', 'v1.2.0')
        self.models['booking_predictor'] = await self.load_model('booking_predictor', 'v1.1.0')
        self.models['entity_extractor'] = await self.load_model('entity_extractor', 'v1.0.5')

    async def predict(self, model_name: str, input_data: Any) -> Prediction:
        model = self.models.get(model_name)
        if not model:
            raise ModelNotFoundError(f"Model {model_name} not found")

        prediction = await model.predict(input_data)
        return prediction
```

## Monitoring & Observability

### ML Model Monitoring
```python
class MLModelMonitor:
    """Monitor ML model performance in production"""

    async def track_prediction_quality(self,
                                     model_name: str,
                                     prediction: Prediction,
                                     actual: Optional[Any] = None):
        """Track prediction quality metrics"""

        # Log prediction
        await self.metrics_store.log_prediction(
            model_name=model_name,
            prediction=prediction,
            timestamp=datetime.utcnow()
        )

        # If actual result available, calculate accuracy
        if actual:
            accuracy = self.calculate_accuracy(prediction, actual)
            await self.metrics_store.log_accuracy(model_name, accuracy)

            # Trigger alerts if accuracy drops
            if accuracy < self.accuracy_threshold:
                await self.alert_service.send_alert(
                    f"Model {model_name} accuracy dropped to {accuracy:.2f}"
                )
```

### Performance Metrics
- **Inference Latency:** Vorhersagezeit pro Request
- **Model Accuracy:** Genauigkeit der Klassifizierung
- **Throughput:** Verarbeitete Dokumente pro Sekunde
- **Resource Usage:** CPU/GPU/Memory Nutzung

## Best Practices

### Model Development
- **Version Control:** Alle Modelle versioniert
- **A/B Testing:** Schrittweise Modell-Rollouts
- **Feature Engineering:** Systematische Merkmalsentwicklung
- **Cross-Validation:** Robuste Modell-Evaluation

### Production Deployment
- **Model Registry:** Zentrale Modell-Verwaltung
- **Canary Deployments:** Risikominimierte Rollouts
- **Fallback Strategies:** Backup-Modelle für Ausfälle
- **Continuous Monitoring:** Produktions-Überwachung

## Lizenz

Proprietary - LLKJJ Team
-- ====================================================================
-- LLKJJ Accounting Software - Database Schema
-- ====================================================================
-- Production-ready PostgreSQL schema for autonomous accounting system
-- Supports: PDF processing, ML results, user feedback, continuous learning
--
-- Author: LLKJJ Accounting Team
-- Version: 1.0.0
-- ====================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ====================================================================
-- USERS & AUTHENTICATION
-- ====================================================================

CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    company_name VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    subscription_plan VARCHAR(50) DEFAULT 'basic',

    -- Accounting settings
    default_skr03_rules JSONB,
    preferred_suppliers JSONB,

    CONSTRAINT valid_email CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_active ON users(is_active);

-- ====================================================================
-- PDF PROCESSING & ML RESULTS
-- ====================================================================

CREATE TABLE invoice_uploads (
    upload_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,

    -- File metadata
    original_filename VARCHAR(500) NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    file_hash VARCHAR(64) UNIQUE NOT NULL, -- SHA-256 for deduplication
    storage_path VARCHAR(1000),

    -- Processing status
    processing_status VARCHAR(20) DEFAULT 'uploaded'
        CHECK (processing_status IN ('uploaded', 'processing', 'completed', 'failed')),

    -- Timestamps
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,

    -- ML processing results
    ml_confidence_score DECIMAL(5,4), -- 0.0000 to 1.0000
    processing_time_ms INTEGER,
    ml_model_version VARCHAR(50),

    -- Error handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,

    CONSTRAINT valid_confidence CHECK (ml_confidence_score IS NULL OR
        (ml_confidence_score >= 0 AND ml_confidence_score <= 1))
);

CREATE INDEX idx_uploads_user_id ON invoice_uploads(user_id);
CREATE INDEX idx_uploads_status ON invoice_uploads(processing_status);
CREATE INDEX idx_uploads_uploaded_at ON invoice_uploads(uploaded_at);
CREATE INDEX idx_uploads_file_hash ON invoice_uploads(file_hash);

-- ====================================================================
-- EXTRACTED INVOICE DATA
-- ====================================================================

CREATE TABLE invoice_headers (
    header_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    upload_id UUID NOT NULL REFERENCES invoice_uploads(upload_id) ON DELETE CASCADE,

    -- Supplier information
    supplier_name VARCHAR(255),
    supplier_id VARCHAR(100), -- Customer number at supplier

    -- Invoice details
    invoice_number VARCHAR(100),
    invoice_date DATE,
    due_date DATE,

    -- Financial totals
    net_amount DECIMAL(15,2),
    tax_amount DECIMAL(15,2),
    tax_rate DECIMAL(5,4),
    gross_amount DECIMAL(15,2),

    -- Additional metadata
    currency_code VARCHAR(3) DEFAULT 'EUR',
    payment_terms VARCHAR(255),
    delivery_date DATE,

    -- ML extraction metadata
    extraction_confidence DECIMAL(5,4),
    manual_corrections JSONB, -- Track user corrections

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_headers_upload_id ON invoice_headers(upload_id);
CREATE INDEX idx_headers_supplier ON invoice_headers(supplier_name);
CREATE INDEX idx_headers_invoice_date ON invoice_headers(invoice_date);

-- ====================================================================
-- INVOICE LINE ITEMS
-- ====================================================================

CREATE TABLE invoice_line_items (
    line_item_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    header_id UUID NOT NULL REFERENCES invoice_headers(header_id) ON DELETE CASCADE,

    -- Line item details
    line_number INTEGER NOT NULL,
    item_description TEXT NOT NULL,
    item_number VARCHAR(100), -- Article/SKU number
    brand VARCHAR(100),

    -- Quantities and pricing
    quantity DECIMAL(12,4) NOT NULL,
    unit_of_measure VARCHAR(20),
    unit_price DECIMAL(15,4),
    line_total DECIMAL(15,2) NOT NULL,

    -- Tax information
    tax_rate DECIMAL(5,4),
    tax_amount DECIMAL(15,2),

    -- Product categorization
    product_category VARCHAR(100),
    elektro_category VARCHAR(100), -- Specialized for electrical trade

    -- ML processing metadata
    extraction_confidence DECIMAL(5,4),
    requires_review BOOLEAN DEFAULT false,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT positive_quantity CHECK (quantity > 0),
    CONSTRAINT positive_line_total CHECK (line_total > 0)
);

CREATE INDEX idx_line_items_header_id ON invoice_line_items(header_id);
CREATE INDEX idx_line_items_line_number ON invoice_line_items(line_number);
CREATE INDEX idx_line_items_product_category ON invoice_line_items(product_category);
CREATE INDEX idx_line_items_requires_review ON invoice_line_items(requires_review);

-- ====================================================================
-- SKR03 CLASSIFICATIONS
-- ====================================================================

CREATE TABLE skr03_classifications (
    classification_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    line_item_id UUID NOT NULL REFERENCES invoice_line_items(line_item_id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(user_id),

    -- SKR03 account classification
    suggested_account VARCHAR(10) NOT NULL, -- AI suggestion
    final_account VARCHAR(10) NOT NULL,     -- User-approved account
    account_description VARCHAR(255),

    -- Classification metadata
    classification_confidence DECIMAL(5,4) NOT NULL,
    classification_method VARCHAR(50) NOT NULL
        CHECK (classification_method IN ('ai_gemini', 'ai_spacy', 'rule_based', 'user_manual')),

    -- User feedback for learning
    was_corrected BOOLEAN DEFAULT false,
    original_suggestion VARCHAR(10), -- Store original AI suggestion if corrected
    correction_reason TEXT,

    -- Matched rules/keywords for transparency
    matched_keywords JSONB,
    applied_rules JSONB,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_accounts CHECK (
        suggested_account ~ '^[0-9]{4}$' AND
        final_account ~ '^[0-9]{4}$'
    )
);

CREATE INDEX idx_classifications_line_item ON skr03_classifications(line_item_id);
CREATE INDEX idx_classifications_user_id ON skr03_classifications(user_id);
CREATE INDEX idx_classifications_final_account ON skr03_classifications(final_account);
CREATE INDEX idx_classifications_was_corrected ON skr03_classifications(was_corrected);

-- ====================================================================
-- MACHINE LEARNING TRAINING DATA
-- ====================================================================

CREATE TABLE ml_training_feedback (
    feedback_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(user_id),
    classification_id UUID REFERENCES skr03_classifications(classification_id),

    -- Feedback type and content
    feedback_type VARCHAR(50) NOT NULL
        CHECK (feedback_type IN ('classification_correction', 'supplier_pattern', 'description_enhancement', 'rule_suggestion')),

    -- Training data
    input_text TEXT NOT NULL, -- Original text that was classified
    expected_output JSONB NOT NULL, -- Correct classification
    model_output JSONB, -- What the model predicted

    -- Feedback quality scoring
    feedback_quality_score DECIMAL(3,2) DEFAULT 1.0, -- 0.0 to 1.0
    training_impact_weight DECIMAL(3,2) DEFAULT 1.0, -- How much this feedback should influence training

    -- Processing status
    applied_to_training BOOLEAN DEFAULT false,
    training_batch_id UUID,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE,

    CONSTRAINT valid_quality_score CHECK (feedback_quality_score >= 0 AND feedback_quality_score <= 1),
    CONSTRAINT valid_impact_weight CHECK (training_impact_weight >= 0 AND training_impact_weight <= 1)
);

CREATE INDEX idx_training_feedback_user_id ON ml_training_feedback(user_id);
CREATE INDEX idx_training_feedback_type ON ml_training_feedback(feedback_type);
CREATE INDEX idx_training_feedback_applied ON ml_training_feedback(applied_to_training);
CREATE INDEX idx_training_feedback_created_at ON ml_training_feedback(created_at);

-- ====================================================================
-- SUPPLIER INTELLIGENCE
-- ====================================================================

CREATE TABLE supplier_patterns (
    pattern_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(user_id), -- NULL = global pattern

    -- Supplier identification
    supplier_name VARCHAR(255) NOT NULL,
    supplier_aliases JSONB, -- Alternative names/spellings

    -- Classification patterns learned from history
    common_accounts JSONB NOT NULL, -- {"3400": 0.85, "4930": 0.15} - account probabilities
    product_categories JSONB NOT NULL, -- Categories this supplier typically provides
    average_confidence DECIMAL(5,4),

    -- Learning metadata
    sample_count INTEGER DEFAULT 0, -- Number of invoices this pattern is based on
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    pattern_quality DECIMAL(3,2) DEFAULT 0.5, -- How reliable this pattern is

    CONSTRAINT positive_sample_count CHECK (sample_count >= 0)
);

CREATE INDEX idx_supplier_patterns_supplier ON supplier_patterns(supplier_name);
CREATE INDEX idx_supplier_patterns_user_id ON supplier_patterns(user_id);
CREATE INDEX idx_supplier_patterns_updated ON supplier_patterns(last_updated);

-- ====================================================================
-- PROCESSING ANALYTICS
-- ====================================================================

CREATE TABLE processing_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(user_id),

    -- Time period
    metric_date DATE NOT NULL,
    metric_hour INTEGER CHECK (metric_hour >= 0 AND metric_hour <= 23),

    -- Processing statistics
    invoices_processed INTEGER DEFAULT 0,
    total_processing_time_ms BIGINT DEFAULT 0,
    average_confidence DECIMAL(5,4),

    -- Quality metrics
    manual_corrections_count INTEGER DEFAULT 0,
    high_confidence_count INTEGER DEFAULT 0, -- confidence > 0.8
    low_confidence_count INTEGER DEFAULT 0,  -- confidence < 0.5

    -- Business impact
    estimated_time_saved_minutes INTEGER DEFAULT 0,
    estimated_cost_savings_euros DECIMAL(10,2) DEFAULT 0,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT positive_counts CHECK (
        invoices_processed >= 0 AND
        manual_corrections_count >= 0 AND
        high_confidence_count >= 0 AND
        low_confidence_count >= 0
    )
);

CREATE INDEX idx_processing_metrics_user_date ON processing_metrics(user_id, metric_date);
CREATE INDEX idx_processing_metrics_date ON processing_metrics(metric_date);

-- ====================================================================
-- VIEWS FOR COMMON QUERIES
-- ====================================================================

-- Complete invoice view with all classifications
CREATE VIEW invoice_complete AS
SELECT
    u.upload_id,
    u.user_id,
    u.original_filename,
    u.processing_status,
    u.ml_confidence_score,

    h.supplier_name,
    h.invoice_number,
    h.invoice_date,
    h.net_amount,
    h.gross_amount,

    COUNT(li.line_item_id) as line_items_count,
    COUNT(CASE WHEN sc.was_corrected = true THEN 1 END) as corrections_count,
    AVG(sc.classification_confidence) as avg_classification_confidence

FROM invoice_uploads u
LEFT JOIN invoice_headers h ON u.upload_id = h.upload_id
LEFT JOIN invoice_line_items li ON h.header_id = li.header_id
LEFT JOIN skr03_classifications sc ON li.line_item_id = sc.line_item_id
GROUP BY u.upload_id, h.header_id;

-- User analytics view
CREATE VIEW user_analytics AS
SELECT
    u.user_id,
    u.company_name,
    COUNT(DISTINCT up.upload_id) as total_invoices,
    AVG(up.ml_confidence_score) as avg_confidence,
    SUM(CASE WHEN sc.was_corrected = true THEN 1 ELSE 0 END) as total_corrections,
    AVG(up.processing_time_ms) as avg_processing_time_ms,
    MAX(up.uploaded_at) as last_upload_date

FROM users u
LEFT JOIN invoice_uploads up ON u.user_id = up.user_id
LEFT JOIN invoice_headers h ON up.upload_id = h.upload_id
LEFT JOIN invoice_line_items li ON h.header_id = li.header_id
LEFT JOIN skr03_classifications sc ON li.line_item_id = sc.line_item_id
GROUP BY u.user_id;

-- ====================================================================
-- FUNCTIONS & TRIGGERS
-- ====================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at trigger to relevant tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_invoice_headers_updated_at BEFORE UPDATE ON invoice_headers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_skr03_classifications_updated_at BEFORE UPDATE ON skr03_classifications
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ====================================================================
-- INITIAL DATA & CONFIGURATION
-- ====================================================================

-- Create default system user for background processes
INSERT INTO users (user_id, email, company_name, is_active)
VALUES (uuid_generate_v4(), 'system@llkjj.local', 'LLKJJ System', true)
ON CONFLICT (email) DO NOTHING;

-- Performance optimization indexes for common queries
CREATE INDEX CONCURRENTLY idx_classifications_confidence_high
ON skr03_classifications(classification_confidence)
WHERE classification_confidence > 0.8;

CREATE INDEX CONCURRENTLY idx_uploads_recent
ON invoice_uploads(uploaded_at DESC)
WHERE uploaded_at > CURRENT_DATE - INTERVAL '30 days';

-- ====================================================================
-- COMMENTS FOR DOCUMENTATION
-- ====================================================================

COMMENT ON TABLE users IS 'User accounts and company settings';
COMMENT ON TABLE invoice_uploads IS 'PDF file uploads and processing status';
COMMENT ON TABLE invoice_headers IS 'Extracted invoice header information';
COMMENT ON TABLE invoice_line_items IS 'Individual line items from invoices';
COMMENT ON TABLE skr03_classifications IS 'German accounting classifications (SKR03)';
COMMENT ON TABLE ml_training_feedback IS 'User feedback for continuous ML learning';
COMMENT ON TABLE supplier_patterns IS 'Learned patterns for supplier-specific classification';
COMMENT ON TABLE processing_metrics IS 'Performance and business metrics tracking';

COMMENT ON VIEW invoice_complete IS 'Complete invoice data with classification summary';
COMMENT ON VIEW user_analytics IS 'Per-user analytics and performance metrics';

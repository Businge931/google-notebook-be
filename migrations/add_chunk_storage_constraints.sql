-- Production Database Constraints for Chunk Storage Integrity
-- Ensures data consistency and prevents orphaned chunks

-- Add constraints to document_chunks table
ALTER TABLE document_chunks 
ADD CONSTRAINT fk_document_chunks_document_id 
FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE;

-- Add index for performance on document_id lookups
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_page_number ON document_chunks(document_id, page_number);

-- Add check constraint to ensure page numbers are valid
ALTER TABLE document_chunks 
ADD CONSTRAINT chk_document_chunks_page_number_positive 
CHECK (page_number > 0);

-- Add check constraint to ensure positions are valid
ALTER TABLE document_chunks 
ADD CONSTRAINT chk_document_chunks_positions_valid 
CHECK (start_position >= 0 AND end_position >= start_position);

-- Add check constraint to ensure text content is not empty
ALTER TABLE document_chunks 
ADD CONSTRAINT chk_document_chunks_text_not_empty 
CHECK (LENGTH(TRIM(text_content)) > 0);

-- Add unique constraint on chunk_id to prevent duplicates
ALTER TABLE document_chunks 
ADD CONSTRAINT uk_document_chunks_chunk_id 
UNIQUE (id);

-- Add constraint to ensure page numbers don't exceed document page count
-- Note: This requires a function to validate against the documents table
CREATE OR REPLACE FUNCTION validate_chunk_page_number()
RETURNS TRIGGER AS $$
BEGIN
    -- Check if page_number exceeds document's page_count
    IF EXISTS (
        SELECT 1 FROM documents 
        WHERE id = NEW.document_id 
        AND page_count IS NOT NULL 
        AND NEW.page_number > page_count
    ) THEN
        RAISE EXCEPTION 'Chunk page number % exceeds document page count', NEW.page_number;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to validate page numbers on insert/update
DROP TRIGGER IF EXISTS trg_validate_chunk_page_number ON document_chunks;
CREATE TRIGGER trg_validate_chunk_page_number
    BEFORE INSERT OR UPDATE ON document_chunks
    FOR EACH ROW
    EXECUTE FUNCTION validate_chunk_page_number();

-- Add constraint to ensure documents have consistent chunk counts
CREATE OR REPLACE FUNCTION update_document_chunk_count()
RETURNS TRIGGER AS $$
BEGIN
    -- Update total_chunks count in documents table
    UPDATE documents 
    SET total_chunks = (
        SELECT COUNT(*) 
        FROM document_chunks 
        WHERE document_id = COALESCE(NEW.document_id, OLD.document_id)
    )
    WHERE id = COALESCE(NEW.document_id, OLD.document_id);
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Create trigger to maintain chunk count consistency
DROP TRIGGER IF EXISTS trg_update_document_chunk_count ON document_chunks;
CREATE TRIGGER trg_update_document_chunk_count
    AFTER INSERT OR UPDATE OR DELETE ON document_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_document_chunk_count();

-- Add logging table for chunk storage operations (for monitoring)
CREATE TABLE IF NOT EXISTS chunk_storage_audit (
    id SERIAL PRIMARY KEY,
    document_id UUID NOT NULL,
    operation VARCHAR(20) NOT NULL, -- 'INSERT', 'UPDATE', 'DELETE', 'VALIDATION_FAILED'
    chunk_count INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunk_storage_audit_document_id ON chunk_storage_audit(document_id);
CREATE INDEX IF NOT EXISTS idx_chunk_storage_audit_created_at ON chunk_storage_audit(created_at);

-- Function to log chunk storage operations
CREATE OR REPLACE FUNCTION log_chunk_storage_operation()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO chunk_storage_audit (document_id, operation, chunk_count)
    VALUES (
        COALESCE(NEW.document_id, OLD.document_id),
        TG_OP,
        (SELECT COUNT(*) FROM document_chunks WHERE document_id = COALESCE(NEW.document_id, OLD.document_id))
    );
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Create audit trigger
DROP TRIGGER IF EXISTS trg_log_chunk_storage_operation ON document_chunks;
CREATE TRIGGER trg_log_chunk_storage_operation
    AFTER INSERT OR UPDATE OR DELETE ON document_chunks
    FOR EACH ROW
    EXECUTE FUNCTION log_chunk_storage_operation();

-- Add view for chunk storage health monitoring
CREATE OR REPLACE VIEW chunk_storage_health AS
SELECT 
    d.id as document_id,
    d.filename,
    d.status,
    d.processing_stage,
    d.page_count,
    d.total_chunks as reported_chunk_count,
    COUNT(dc.id) as actual_chunk_count,
    CASE 
        WHEN d.total_chunks = COUNT(dc.id) THEN 'CONSISTENT'
        WHEN d.total_chunks IS NULL OR COUNT(dc.id) = 0 THEN 'NO_CHUNKS'
        ELSE 'INCONSISTENT'
    END as consistency_status,
    MAX(dc.page_number) as max_chunk_page,
    CASE 
        WHEN d.page_count IS NOT NULL AND MAX(dc.page_number) > d.page_count THEN 'PAGE_OVERFLOW'
        WHEN d.page_count IS NULL THEN 'NO_PAGE_COUNT'
        ELSE 'PAGE_VALID'
    END as page_validation_status,
    d.updated_at as last_document_update,
    MAX(csa.created_at) as last_chunk_operation
FROM documents d
LEFT JOIN document_chunks dc ON d.id = dc.document_id
LEFT JOIN chunk_storage_audit csa ON d.id::text = csa.document_id::text
GROUP BY d.id, d.filename, d.status, d.processing_stage, d.page_count, d.total_chunks, d.updated_at
ORDER BY d.updated_at DESC;

-- Grant necessary permissions
GRANT SELECT ON chunk_storage_health TO PUBLIC;
GRANT SELECT ON chunk_storage_audit TO PUBLIC;

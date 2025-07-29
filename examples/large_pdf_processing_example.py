"""
Example script demonstrating large PDF processing with the new async API.

This shows how to:
1. Upload a large PDF
2. Start async processing 
3. Monitor progress in real-time
4. Handle completion/errors
"""
import asyncio
import aiohttp
import json
import time
from typing import Optional

class LargePDFProcessor:
    """Client for processing large PDFs using the async API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def upload_pdf(self, file_path: str) -> str:
        """Upload PDF and return document_id."""
        print(f"📤 Uploading PDF: {file_path}")
        
        with open(file_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=file_path.split('/')[-1], content_type='application/pdf')
            
            async with self.session.post(f"{self.base_url}/api/v1/documents/upload", data=data) as response:
                if response.status != 201:
                    error_text = await response.text()
                    raise Exception(f"Upload failed: {response.status} - {error_text}")
                
                result = await response.json()
                document_id = result['document_id']
                print(f"✅ Upload successful! Document ID: {document_id}")
                return document_id
    
    async def start_async_processing(self, document_id: str, force_reprocess: bool = False) -> str:
        """Start async processing and return job_id."""
        print(f"🚀 Starting async processing for document: {document_id}")
        
        url = f"{self.base_url}/api/v1/documents/{document_id}/process-async"
        params = {"force_reprocess": force_reprocess}
        
        async with self.session.post(url, params=params) as response:
            if response.status != 202:
                error_text = await response.text()
                raise Exception(f"Processing start failed: {response.status} - {error_text}")
            
            result = await response.json()
            job_id = result['job_id']
            print(f"✅ Processing started! Job ID: {job_id}")
            print(f"📊 Status URL: {result['status_url']}")
            print(f"🔌 WebSocket URL: {result['websocket_url']}")
            return job_id
    
    async def get_processing_status(self, document_id: str, job_id: str) -> dict:
        """Get current processing status."""
        url = f"{self.base_url}/api/v1/documents/{document_id}/processing-status/{job_id}"
        
        async with self.session.get(url) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Status check failed: {response.status} - {error_text}")
            
            return await response.json()
    
    async def monitor_processing(self, document_id: str, job_id: str, poll_interval: int = 5) -> dict:
        """Monitor processing until completion."""
        print(f"👀 Monitoring processing progress...")
        
        start_time = time.time()
        last_progress = None
        
        while True:
            try:
                status = await self.get_processing_status(document_id, job_id)
                
                # Print status updates
                job_status = status['status']
                print(f"\n📊 Status: {job_status.upper()}")
                
                if 'progress' in status:
                    progress = status['progress']
                    
                    # Only print if progress changed
                    if progress != last_progress:
                        print(f"   📄 Pages: {progress['processed_pages']}/{progress['total_pages']}")
                        print(f"   📝 Chunks: {progress['processed_chunks']}/{progress['total_chunks']}")
                        print(f"   🔢 Vectorized: {progress['vectorized_chunks']}/{progress['total_chunks']}")
                        print(f"   🎯 Stage: {progress['current_stage']}")
                        print(f"   💾 Memory: {progress['memory_usage_mb']:.1f}MB")
                        print(f"   ⏱️  Time: {progress['processing_time_ms']}ms")
                        
                        if 'completion_percentage' in progress:
                            print(f"   📈 Progress: {progress['completion_percentage']:.1f}%")
                        
                        if progress.get('estimated_remaining_ms'):
                            remaining_sec = progress['estimated_remaining_ms'] / 1000
                            print(f"   ⏳ ETA: {remaining_sec:.1f}s")
                        
                        last_progress = progress
                
                # Check if completed
                if job_status in ['completed', 'failed', 'cancelled']:
                    elapsed_time = time.time() - start_time
                    print(f"\n🏁 Processing {job_status.upper()} in {elapsed_time:.1f}s")
                    
                    if job_status == 'failed' and status.get('error_message'):
                        print(f"❌ Error: {status['error_message']}")
                    
                    return status
                
                # Wait before next poll
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                print(f"❌ Error checking status: {e}")
                await asyncio.sleep(poll_interval)
    
    async def process_large_pdf_complete(self, file_path: str, force_reprocess: bool = False) -> dict:
        """Complete workflow: upload + process + monitor."""
        print(f"🎯 Starting complete large PDF processing workflow")
        print(f"📁 File: {file_path}")
        print(f"🔄 Force reprocess: {force_reprocess}")
        print("=" * 60)
        
        try:
            # Step 1: Upload
            document_id = await self.upload_pdf(file_path)
            
            # Step 2: Start processing
            job_id = await self.start_async_processing(document_id, force_reprocess)
            
            # Step 3: Monitor until completion
            final_status = await self.monitor_processing(document_id, job_id)
            
            print("=" * 60)
            print(f"🎉 Workflow completed!")
            print(f"📄 Document ID: {document_id}")
            print(f"🆔 Job ID: {job_id}")
            print(f"✅ Final Status: {final_status['status'].upper()}")
            
            return {
                'document_id': document_id,
                'job_id': job_id,
                'final_status': final_status
            }
            
        except Exception as e:
            print(f"💥 Workflow failed: {e}")
            raise


async def main():
    """Example usage of the large PDF processor."""
    
    # Configuration
    PDF_FILE = "50mb.pdf"  # Update this path to your test file
    BASE_URL = "http://localhost:8000"
    
    print("🚀 Large PDF Processing Example")
    print(f"🌐 API Base URL: {BASE_URL}")
    print(f"📄 PDF File: {PDF_FILE}")
    print("=" * 60)
    
    async with LargePDFProcessor(BASE_URL) as processor:
        try:
            # Process the large PDF
            result = await processor.process_large_pdf_complete(
                file_path=PDF_FILE,
                force_reprocess=True  # Set to True to reprocess existing files
            )
            
            print("\n🎊 SUCCESS! Large PDF processed successfully!")
            print(f"📋 Result: {json.dumps(result, indent=2)}")
            
        except FileNotFoundError:
            print(f"❌ File not found: {PDF_FILE}")
            print("💡 Please ensure the 50mb.pdf file exists in the current directory")
            
        except Exception as e:
            print(f"💥 Processing failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())

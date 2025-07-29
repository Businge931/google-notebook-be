"""
Netlify Function for FastAPI backend - Python version
"""
import os
import sys
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables for production
os.environ.setdefault("DEBUG", "false")
os.environ.setdefault("STORAGE_TYPE", "local")
os.environ.setdefault("UPLOAD_DIR", "/tmp/uploads")

try:
    from fastapi import FastAPI, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    
    # Create a minimal FastAPI app
    app = FastAPI()
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Import and include the main router
    from main import app as main_app
    app.include_router(main_app.router)
    
    # Handler for Netlify Functions
    def handler(event, context):
        # Convert API Gateway event to ASGI scope
        from mangum import Mangum
        handler = Mangum(app, lifespan="off")
        response = handler(event, context)
        return response
    
    # For local testing
    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
except ImportError as e:
    print(f"Import error: {e}")
    
    def handler(event, context):
        return {
            "statusCode": 500,
            "body": json.dumps({
                "status": "error",
                "message": "Failed to initialize FastAPI application",
                "details": str(e)
            }),
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*"
            }
        }

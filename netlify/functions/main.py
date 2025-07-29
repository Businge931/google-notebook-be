"""
Netlify Function for FastAPI backend
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

# Try to import FastAPI app and Mangum
try:
    from main import app
    from mangum import Mangum
    
    # Create the handler for Netlify Functions
    handler = Mangum(
        app,
        lifespan="off",
        api_gateway_base_path="/.netlify/functions/main"
    )
    
except ImportError as e:
    print(f"Import error: {e}")
    
    # Fallback handler with better error reporting
    def handler(event, context):
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*"
            },
            "body": json.dumps({
                "status": "error",
                "message": "Failed to initialize FastAPI application",
                "error": str(e)
            })
        }

# Netlify Functions requires a lambda_handler function
def lambda_handler(event, context):
    return handler(event, context)

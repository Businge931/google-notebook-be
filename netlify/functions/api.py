"""
Netlify Function for FastAPI backend
"""
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables for production
os.environ.setdefault("DEBUG", "false")
os.environ.setdefault("STORAGE_TYPE", "local")
os.environ.setdefault("UPLOAD_DIR", "/tmp/uploads")

try:
    from main import app
    from mangum import Mangum
    
    # Create the handler for Netlify Functions
    handler = Mangum(app, lifespan="off", api_gateway_base_path="/api/v1")
    
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback handler
    def handler(event, context):
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": f'{{"error": "Import error: {str(e)}"}}'
        }

# Export the handler
def lambda_handler(event, context):
    return handler(event, context)

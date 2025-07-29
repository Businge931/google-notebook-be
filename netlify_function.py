"""
Netlify Functions wrapper for FastAPI application
"""
import os
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Set environment variables for production
os.environ.setdefault("DEBUG", "false")
os.environ.setdefault("STORAGE_TYPE", "local")
os.environ.setdefault("UPLOAD_DIR", "/tmp/uploads")

try:
    from main import app
    from mangum import Mangum
    
    # Create the handler for Netlify Functions
    handler = Mangum(app, lifespan="off")
    
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback handler
    def handler(event, context):
        return {
            "statusCode": 500,
            "body": f"Import error: {str(e)}"
        }

# Export for Netlify
def main(event, context):
    return handler(event, context)

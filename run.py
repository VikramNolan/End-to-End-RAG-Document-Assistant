import sys
from pathlib import Path
import uvicorn

# Ensure the Backend folder is in Python's import search path
PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

def main():
    print("=" * 60)
    print("Starting RAG Assistant Server with uv...")
    print("-> Web Application: http://127.0.0.1:8000")
    print("-> API Documentation: http://127.0.0.1:8000/docs")
    print("=" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        app_dir=str(BACKEND_DIR)
    )

if __name__ == "__main__":
    main()

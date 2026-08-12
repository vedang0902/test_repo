"""
Entry point for PaymentPipeline server.
Run with: python run.py
Or:        uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
"""
import uvicorn
from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.api_port,
        log_config=None,          # Use our own logging config
        access_log=False,         # Handled by middleware
        reload=False,             # Disable in production
        workers=1,                # Single worker to preserve in-memory bug state
    )

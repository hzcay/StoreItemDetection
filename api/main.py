from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, List
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime
import sys
from pathlib import Path

from database import engine, get_db
from models import Base

# Import routers
from controllers.product_controller import router as product_router
from controllers.categories_controller import router as categories_router
from qdrant_utils.qdrant_client import initialize_model, create_qdrant_client

# Create database tables
Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="Store Item Detection API",
    description="API for managing store items, products, and categories",
    version="1.0.0",
    docs_url=None,  
    redoc_url=None,  
    openapi_url="/api/openapi.json"
)

# Custom Swagger UI
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/api/openapi.json",
        title="Store Item Detection API - Swagger UI",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Store Item Detection API",
        version="1.0.0",
        description="Comprehensive API for managing store items, products, and categories",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(product_router, tags=["Products"])
app.include_router(categories_router, tags=["Categories"])

# Health check endpoint
@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Check if the API is running and healthy",
    response_description="API status",
    responses={
        200: {
            "description": "API is running",
            "content": {
                "application/json": {
                    "example": {"status": "ok", "message": "API is running"}
                }
            }
        }
    }
)
async def health_check():
    return {"status": "ok", "message": "API is running"}

# Test database connection
@app.get(
    "/test-db",
    summary="Test Database Connection",
    description="Test the database connection and return the current timestamp",
    responses={
        200: {
            "description": "Database connection successful",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "Database connection successful",
                        "timestamp": "2025-11-25T10:00:00.000Z"
                    }
                }
            }
        }
    }
)
async def test_db_connection(db: Session = Depends(get_db)):
    try:
        # Try to execute a simple query
        db.execute(text("SELECT 1"))
        return {
            "status": "success",
            "message": "Database connection successful",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to connect to the database",
                "error": str(e)
            }
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)     
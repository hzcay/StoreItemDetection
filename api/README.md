# Store Item Detection API

FastAPI-based API service for store item detection.

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Unix or MacOS
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Development
```bash
uvicorn main:app --reload
```

The API will be available at http://localhost:8000

### Production
For production, use a production-ready server like Uvicorn with Gunicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Environment Variables
Create a `.env` file in the project root with the following variables:
```
ENVIRONMENT=development
DEBUG=True
API_PREFIX=/api
HOST=0.0.0.0
PORT=8000
```

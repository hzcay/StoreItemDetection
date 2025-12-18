# api/controllers/product_controller.py
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query, Path, Form
from typing import List
from sqlalchemy.orm import Session
import os
import uuid
from typing import Optional

from qdrant_utils.qdrant_client import (
    create_qdrant_client,  # kept for compatibility
    initialize_model,      # kept for compatibility
    save_embedding_to_qdrant,
    search_similar,
    get_model,
    get_qdrant_client,
)
from database import get_db
from schemas.schema import ProductCreate, ProductResponse, ProductUpdate, ProductSearchResult
from services.product_service import ProductService

router = APIRouter(
    prefix="/api/products",
    tags=["Products"],
    responses={
        404: {"description": "Product not found"},
        500: {"description": "Internal server error"}
    }
)

@router.post(
    "/",
    response_model=ProductResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new product with images"
)
def create_product(
    name: str = Form(...),
    description: str = Form(None),
    price: float = Form(...),
    stock_quantity: int = Form(0),
    sku: str = Form(None),
    barcode: str = Form(None),
    is_active: bool = Form(True),
    category_id: int = Form(None),
    images: List[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    print("Creating product...")
    print("Price: ", price)
    print("Stock quantity: ", stock_quantity)
    print("SKU: ", sku)
    print("Barcode: ", barcode)
    print("Is active: ", is_active)
    print("Category ID: ", category_id)
    # ------------------------------
    # Validate price
    # ------------------------------
    if price <= 0:
        raise HTTPException(status_code=400, detail="Price must be greater than 0")

    service = ProductService(db)

    # Check duplicate SKU
    if sku and service.product_repo.get_product_by_sku(sku):
        raise HTTPException(status_code=400, detail=f"SKU '{sku}' already exists")

    # ------------------------------
    # Create product record
    # ------------------------------
    product_data = ProductCreate(
        name=name,
        description=description,
        price=price,
        stock_quantity=stock_quantity,
        sku=sku or str(uuid.uuid4()),
        barcode=barcode or str(uuid.uuid4()),
        is_active=is_active,
        category_id=category_id
    )
    print("Product data created successfully")
    product = service.create_product(product_data)
    print("Product created successfully")
    qdrant_client = get_qdrant_client()
    model = get_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not initialized")
    if qdrant_client is None:
        raise HTTPException(status_code=500, detail="Qdrant client is not initialized")
    else:
        print("Model and Qdrant client initialized successfully")
    # ------------------------------
    # Save images + generate embeddings
    # ------------------------------
    if images:
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)

        for idx, image in enumerate(images):
            try:
                # Validate extension
                ext = image.filename.split('.')[-1].lower()
                if ext not in {"png", "jpg", "jpeg", "gif", "webp"}:
                    print(f"Skipping unsupported file: {image.filename}")
                    continue

                # Store the image
                filename = f"{uuid.uuid4()}.{ext}"
                file_path = os.path.join(upload_dir, filename)

                with open(file_path, "wb") as f:
                    f.write(image.file.read())

                # Add DB image record
                is_primary = idx == 0
                service.add_product_image(product.id, file_path, is_primary)

                # Save embedding to Qdrant
                save_embedding_to_qdrant(
                    model,
                    qdrant_client,
                    product_id=product.id,
                    image_path=file_path,
                    point_id=str(uuid.uuid4())  # Qdrant expects unsigned int or UUID
                )

            except Exception as e:
                print(f"Error uploading image {image.filename}: {str(e)}")
                continue

    return product



@router.post(
    "/search",
    response_model=List[ProductResponse]
)
def search_product(
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    service = ProductService(db)
    return service.handleSearchProductByImage(image)
# ------------------------------
# Get Product by ID
# ------------------------------
@router.get(
    "/{product_id}",
    response_model=ProductResponse,
    summary="Get product by ID"
)
def get_product(
    product_id: int = Path(..., description="The ID of the product to retrieve", example=1),
    db: Session = Depends(get_db)
):
    service = ProductService(db)
    product = service.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail=f"Product with ID {product_id} not found")
    return product

# ------------------------------
# List Products
# ------------------------------
@router.get(
    "/",
    response_model=List[ProductResponse],
    summary="List all products"
)
def list_products(
    skip: int = Query(0, description="Number of records to skip", ge=0),
    limit: int = Query(100, description="Max number of records to return", le=1000),
    db: Session = Depends(get_db)
):
    service = ProductService(db)
    return service.list_products(skip=skip, limit=limit)




# ------------------------------
# Search similar products by image with ProductSearchResult
# ------------------------------
@router.post(
        "/search-by-image",
        response_model=List[ProductSearchResult],
    summary="Search products by image using embedding similarity",
    description="Upload an image, returns top-K similar products from Qdrant with similarity threshold and scores."
)
async def search_product_by_image(
    image: UploadFile = File(..., description="Query image file"),
    k: int = Query(20, ge=1, le=100, description="Number of results to return (top-K)"),
    threshold: float = Query(0.0, ge=0.0, le=1.0, description="Minimum similarity score threshold"),
    db: Session = Depends(get_db)
):

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    ext = image.filename.split(".")[-1].lower()
    filename = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join(upload_dir, filename)
    
    try:
        with open(file_path, "wb") as f:
            f.write(await image.read())
        
        # Get model and Qdrant client
        model = get_model()
        client = get_qdrant_client()
        if model is None or client is None:
            raise HTTPException(status_code=500, detail="Model or Qdrant client not initialized")
        
        # Search similar embeddings
        hits = search_similar(model, client, file_path, top_k=k)
        
        # Filter by threshold and get product IDs
        filtered_hits = [hit for hit in hits if hit.get("score", 0.0) >= threshold]
        
        service = ProductService(db)
        results = []
        seen_product_ids = set()
        
        for hit in filtered_hits:
            product_id = hit.get("payload", {}).get("product_id")
            if not product_id or product_id in seen_product_ids:
                continue
            
            try:
                product = service.get_product(product_id)
                seen_product_ids.add(product_id)
                
                # Get similarity score
                score = hit.get("score", 0.0)
                similarity_percent = score * 100.0
                
                results.append(ProductSearchResult(
                    product=product,
                    score=score,
                    similarity_percent=similarity_percent
                ))
            except HTTPException:
                # Product not found in DB, skip
                continue
        
        # Sort by score descending (highest similarity first)
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    finally:
        # Clean up temp file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass


# ------------------------------
# Update Product
# ------------------------------
@router.put(
    "/{product_id}",
    response_model=ProductResponse,
    summary="Update a product"
)
def update_product(
    product_id: int = Path(..., description="The ID of the product to update", example=1),
    product_update: ProductUpdate = None,
    db: Session = Depends(get_db)
):
    if product_update is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No update data provided"
        )
    service = ProductService(db)
    updated_product = service.update_product(product_id, product_update)
    if not updated_product:
        raise HTTPException(status_code=404, detail=f"Product with ID {product_id} not found")
    return updated_product

# ------------------------------
# Delete Product
# ------------------------------
@router.delete(
    "/{product_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a product"
)
def delete_product(
    product_id: int = Path(..., description="The ID of the product to delete", example=1),
    db: Session = Depends(get_db)
):
    service = ProductService(db)
    deleted = service.delete_product(product_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Product with ID {product_id} not found")
    return None

# ------------------------------
# Upload Product Image
# ------------------------------
@router.post(
    "/{product_id}/upload-image/",
    status_code=status.HTTP_201_CREATED,
    summary="Upload product image"
)
def upload_product_image(
    product_id: int = Path(..., description="The ID of the product to upload image for", example=1),
    file: UploadFile = File(..., description="Image file to upload"),
    is_primary: bool = Query(False, description="Set as primary image"),
    db: Session = Depends(get_db)
):
    # Validate file type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    file_extension = file.filename.split('.')[-1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Create upload directory
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Generate unique filename
    file_name = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(upload_dir, file_name)
    
    try:
        # Save file synchronously
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())
        
        # Save file info to DB if needed
        service = ProductService(db)
        service.add_product_image(product_id, file_path, is_primary)
        
        return {
            "status": "success",
            "message": "Image uploaded successfully",
            "file_path": file_path,
            "is_primary": is_primary
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading file: {str(e)}"
        )

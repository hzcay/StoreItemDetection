# api/controllers/product_controller.py
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query, Path, Form
from typing import List
from sqlalchemy.orm import Session
import os
import uuid
from typing import Optional
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

    # Load YOLO model
    base_dir = os.path.dirname(__file__) 
    yolo_model_path = os.path.join(base_dir, "..", "models", "best_new_15_12.pt")
    yolo_model = YOLO(yolo_model_path)

    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return []

    results = yolo_model(img)
    detections = results[0]

    # Get embedding model and Qdrant client
    embedding_model = get_model()
    client = get_qdrant_client()
    if embedding_model is None or client is None:
        raise HTTPException(status_code=500, detail="Model or Qdrant client not initialized")
    
    # Create a copy of the image to draw bounding boxes on
    img_with_boxes = img.copy()
    
    # Temporary directory for crop images
    temp_crops_dir = os.path.join(upload_dir, "temp_crops")
    os.makedirs(temp_crops_dir, exist_ok=True)
    
    service = ProductService(db)
    all_results = []
    seen_product_ids = set()
    
    try:
        # Save uploaded image
        with open(file_path, "wb") as f:
            f.write(image_bytes)

        for i, box in enumerate(detections.boxes):
            # get xyxy bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0]) if hasattr(box, 'conf') and len(box.conf) > 0 else 0.0

            # crop the region from the original image
            crop = img[y1:y2, x1:x2]

            # skip invalid crops
            if crop.size == 0:
                continue

            # Save crop temporarily
            crop_filename = f"crop_{i}_{uuid.uuid4()}.jpg"
            crop_path = os.path.join(temp_crops_dir, crop_filename)
            cv2.imwrite(crop_path, crop)
            
            # Initialize label
            label = None
            
            try:
                # Search similar products for this crop in Qdrant
                hits = search_similar(embedding_model, client, crop_path, top_k=k)
                
                # Filter by threshold and get the best match
                filtered_hits = [hit for hit in hits if hit.get("score", 0.0) >= threshold]
                
                if filtered_hits:
                    # Get the best matching product
                    best_hit = filtered_hits[0]
                    product_id = best_hit.get("payload", {}).get("product_id")
                    best_score = best_hit.get("score", 0.0)
                    
                    if product_id:
                        try:
                            product = service.get_product(product_id)
                            
                            # Add to results if not seen before
                            if product_id not in seen_product_ids:
                                seen_product_ids.add(product_id)
                                all_results.append(ProductSearchResult(
                                    product=product,
                                    score=best_score,
                                    similarity_percent=best_score * 100.0
                                ))
                            
                            # Prepare label for bounding box
                            max_label_length = 25
                            product_name = product.name
                            if len(product_name) > max_label_length:
                                product_name = product_name[:max_label_length-3] + "..."
                            label = f"{product_name} ({best_score:.2f})"
                        except HTTPException:
                            label = None
                
            except Exception as e:
                print(f"Error processing crop {i}: {str(e)}")
                label = None
            
            finally:
                # Clean up crop file
                try:
                    if os.path.exists(crop_path):
                        os.remove(crop_path)
                except OSError:
                    pass
            
            # Draw bounding box
            color = (0, 255, 0)  # Green color in BGR
            thickness = 2
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)
            
            # Only add label if product was found
            if label:
                # Add label with product prediction
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                label_thickness = 1
                (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, label_thickness)
                
                # Calculate label position - ensure it doesn't go outside image bounds
                img_height, img_width = img_with_boxes.shape[:2]
                label_x = x1
                label_y = y1 - 5
                
                # If label would go above image, place it below the box
                if label_y - label_height < 0:
                    label_y = y2 + label_height + 5
                    # If still outside, place inside box at top
                    if label_y + label_height > img_height:
                        label_y = y1 + label_height + 5
                
                # Ensure label doesn't go beyond image width
                if label_x + label_width > img_width:
                    label_x = img_width - label_width - 5
                    if label_x < 0:
                        label_x = 5
                
                # Draw background rectangle for text
                bg_x1 = label_x - 2
                bg_y1 = label_y - label_height - 2
                bg_x2 = label_x + label_width + 2
                bg_y2 = label_y + baseline + 2
                
                # Ensure background is within image bounds
                if bg_y1 < 0:
                    bg_y1 = 0
                if bg_y2 > img_height:
                    bg_y2 = img_height
                if bg_x1 < 0:
                    bg_x1 = 0
                if bg_x2 > img_width:
                    bg_x2 = img_width
                
                cv2.rectangle(img_with_boxes, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
                
                # Draw text
                cv2.putText(img_with_boxes, label, (label_x, label_y), 
                           font, font_scale, (0, 0, 0), label_thickness)
        
        # Display the image with bounding boxes using matplotlib
        img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(15, 10))
        plt.imshow(img_with_boxes_rgb)
        plt.title(f'Search Image with Product Predictions ({len(detections.boxes)} objects detected)')
        plt.axis('off')
        plt.show()
        
        # Sort by score descending (highest similarity first)
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results
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

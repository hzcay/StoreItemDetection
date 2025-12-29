# api/controllers/product_controller.py
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query, Path, Form
from typing import List
from sqlalchemy.orm import Session
import os
import uuid
import io
from typing import Optional
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

from api.qdrant_utils.qdrant_client import (
    create_qdrant_client,  # kept for compatibility
    initialize_model,      # kept for compatibility
    save_embedding_to_qdrant,
    search_similar,
    get_model,
    get_qdrant_client,
)
from api.database import get_db
from api.schemas.schema import ProductCreate, ProductResponse, ProductUpdate, ProductSearchResult, ProductSearchResponse
from api.services.product_service import ProductService
from api.services.minio_client import minio_client, MINIO_BUCKET

# Cache YOLO model để không load lại mỗi request
_YOLO_MODEL_CACHE = None

def get_yolo_model():
    """Get YOLO model với caching - ưu tiên custom model đã train"""
    global _YOLO_MODEL_CACHE
    if _YOLO_MODEL_CACHE is not None:
        return _YOLO_MODEL_CACHE
    
    # Ưu tiên dùng custom model đã train trước
    try:
        base_dir = os.path.dirname(__file__) 
        yolo_model_path = os.path.join(base_dir, "..", "models", "best_new_15_12.pt")
        if os.path.exists(yolo_model_path):
            _YOLO_MODEL_CACHE = YOLO(yolo_model_path)
            print(f"✅ Using custom YOLO model: {yolo_model_path} (cached)")
            return _YOLO_MODEL_CACHE
    except Exception as e:
        print(f"⚠️  Custom model not found, trying pretrained: {e}")
    
    # Fallback về pretrained models nếu không có custom
    model_options = [
        'yolo11x.pt',  # YOLOv11 xlarge - fallback
        'yolo11l.pt',  # YOLOv11 large - fallback
        'yolo8x.pt',   # YOLOv8 xlarge - fallback
        'yolo8l.pt',   # YOLOv8 large - fallback
    ]
    
    for model_name in model_options:
        try:
            _YOLO_MODEL_CACHE = YOLO(model_name)
            print(f"✅ Using {model_name} pretrained model (cached)")
            return _YOLO_MODEL_CACHE
        except Exception as e:
            continue
    
    raise HTTPException(status_code=500, detail="No YOLO model available. Please ensure custom model exists or ultralytics can download pretrained models.")

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
        for idx, image in enumerate(images):
            try:
                ext = image.filename.split(".")[-1].lower()
                if ext not in {"png", "jpg", "jpeg", "gif", "webp"}:
                    continue

                object_name = f"products/{product.id}/{uuid.uuid4()}.{ext}"
                image_bytes = image.file.read()

                # ---- Upload to MinIO ----
                minio_client.put_object(
                    bucket_name=MINIO_BUCKET,
                    object_name=object_name,
                    data=io.BytesIO(image_bytes),
                    length=len(image_bytes),
                    content_type=image.content_type,
                )

                is_primary = idx == 0
                object_key = object_name

                service.add_product_image(
                    product.id,
                    object_key,
                    is_primary
                )

                # ---- Save embedding ----
                save_embedding_to_qdrant(
                    model=model,
                    client=qdrant_client,
                    product_id=product.id,
                    image_bytes=image_bytes,
                    additional_data={
                        "object_key": object_name,
                        "is_primary": is_primary
                    },
                    point_id=str(uuid.uuid4())
                )

            except Exception as e:
                print(f"❌ Error uploading image {image.filename}: {e}")


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
    response_model=ProductSearchResponse,
    summary="Search products by image (with YOLO auto-detection or manual crop)"
)
async def search_product_by_image(
    image: UploadFile = File(..., description="Query image file"),
    k: int = Query(10, ge=1, le=100, description="Number of results to return (top-K)"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Minimum similarity score threshold"),
    visual_weight: float = Query(0.6, ge=0.0, le=1.0, description="Weight for visual similarity (default: 0.8)"),
    color_weight: float = Query(0.4, ge=0.0, le=1.0, description="Weight for color similarity (default: 0.2)"),
    use_yolo: bool = Query(True, description="Use YOLO for auto-detection, if False will search entire image"),
    db: Session = Depends(get_db)
):
    # Read image bytes
    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return ProductSearchResponse(
            results=[],
            suggested_products=[],
            has_exact_match=False
        )

    # -------- Embedding infra --------
    model = get_model()
    client = get_qdrant_client()
    if model is None or client is None:
        raise HTTPException(500, "Embedding model or Qdrant not initialized")

    service = ProductService(db)
    seen_product_ids = set()
    exact_results = []  # Results >= threshold (best match per detection)
    all_suggested = []  # All top-K results >= threshold (for suggestions)

    # Mode 1: Use YOLO for auto-detection
    if use_yolo:
        # Load YOLO model (cached) - ưu tiên custom model đã train
        yolo_model = get_yolo_model()
        
        # Run detection với confidence threshold và NMS tốt hơn
        # conf=0.25: chỉ lấy detections có confidence >= 25%
        # iou=0.45: IoU threshold cho NMS (Non-Maximum Suppression)
        results = yolo_model(img, conf=0.25, iou=0.45, verbose=False)
        detections = results[0]

        # Process each detected object
        for i, box in enumerate(detections.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Convert crop to bytes (JPEG format)
            try:
                _, crop_encoded = cv2.imencode('.jpg', crop)
                crop_bytes = crop_encoded.tobytes()
            except Exception as e:
                print(f"Error encoding crop {i}: {e}")
                continue

            # Search similar products for this crop in Qdrant
            try:
                hits = search_similar(
                    model=model,
                    client=client,
                    image_bytes=crop_bytes,
                    top_k=k,
                    visual_weight=visual_weight,
                    color_weight=color_weight
                )
                
                # Separate exact matches (>= threshold) and all suggestions
                filtered_hits = [hit for hit in hits if hit.get("score", 0.0) >= threshold]
                
                # Process exact matches
                if filtered_hits:
                    best_hit = filtered_hits[0]
                    product_id = best_hit.get("payload", {}).get("product_id")
                    best_score = best_hit.get("score", 0.0)
                    visual_score = best_hit.get("visual_score", 0.0)
                    color_score = best_hit.get("color_score", 0.0)
                    
                    if product_id and product_id not in seen_product_ids:
                        try:
                            product = service.get_product(product_id)
                            seen_product_ids.add(product_id)
                            exact_results.append(ProductSearchResult(
                                product=product,
                                score=best_score,
                                similarity_percent=best_score * 100.0,
                                bbox={"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                                visual_score=visual_score,
                                color_score=color_score
                            ))
                        except HTTPException:
                            pass
                
                # Process all suggestions (top-K, but also >= threshold)
                for hit in hits[:k]:  # Take top K
                    product_id = hit.get("payload", {}).get("product_id")
                    score = hit.get("score", 0.0)
                    visual_score = hit.get("visual_score", 0.0)
                    color_score = hit.get("color_score", 0.0)
                    
                    # Only include suggestions that are >= threshold
                    if score >= threshold and product_id and product_id not in seen_product_ids:
                        try:
                            product = service.get_product(product_id)
                            seen_product_ids.add(product_id)
                            all_suggested.append(ProductSearchResult(
                                product=product,
                                score=score,
                                similarity_percent=score * 100.0,
                                bbox={"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                                visual_score=visual_score,
                                color_score=color_score
                            ))
                        except HTTPException:
                            pass
            
            except Exception as e:
                print(f"Error processing crop {i}: {str(e)}")
                continue
    
    # Mode 2: Search entire image (no YOLO, user will crop manually)
    else:
        # Use entire image for search
        try:
            _, img_encoded = cv2.imencode('.jpg', img)
            img_bytes = img_encoded.tobytes()
            
            hits = search_similar(
                model=model,
                client=client,
                image_bytes=img_bytes,
                top_k=k,
                visual_weight=visual_weight,
                color_weight=color_weight
            )
            
            # Filter by threshold
            filtered_hits = [hit for hit in hits if hit.get("score", 0.0) >= threshold]
            
            # Process exact matches
            if filtered_hits:
                best_hit = filtered_hits[0]
                product_id = best_hit.get("payload", {}).get("product_id")
                best_score = best_hit.get("score", 0.0)
                visual_score = best_hit.get("visual_score", 0.0)
                color_score = best_hit.get("color_score", 0.0)
                
                if product_id:
                    try:
                        product = service.get_product(product_id)
                        exact_results.append(ProductSearchResult(
                            product=product,
                            score=best_score,
                            similarity_percent=best_score * 100.0,
                            bbox=None,  # No bbox for manual crop
                            visual_score=visual_score,
                            color_score=color_score
                        ))
                    except HTTPException:
                        pass
            
            # Process suggestions
            for hit in hits[:k]:
                product_id = hit.get("payload", {}).get("product_id")
                score = hit.get("score", 0.0)
                visual_score = hit.get("visual_score", 0.0)
                color_score = hit.get("color_score", 0.0)
                
                if score >= threshold and product_id and product_id not in [r.product.id for r in exact_results]:
                    try:
                        product = service.get_product(product_id)
                        all_suggested.append(ProductSearchResult(
                            product=product,
                            score=score,
                            similarity_percent=score * 100.0,
                            bbox=None,
                            visual_score=visual_score,
                            color_score=color_score
                        ))
                    except HTTPException:
                        pass
        
        except Exception as e:
            print(f"Error processing image: {str(e)}")

    # Sort by score descending
    exact_results.sort(key=lambda x: x.score, reverse=True)
    all_suggested.sort(key=lambda x: x.score, reverse=True)
    
    # Remove duplicates from suggested (if already in exact_results)
    exact_product_ids = {r.product.id for r in exact_results}
    suggested_products = [r for r in all_suggested if r.product.id not in exact_product_ids]
    
    return ProductSearchResponse(
        results=exact_results,
        suggested_products=suggested_products[:k],  # Limit suggested to top K
        has_exact_match=len(exact_results) > 0
    )


@router.post(
    "/search-by-cropped-image",
    response_model=ProductSearchResponse,
    summary="Search products by manually cropped image"
)
async def search_product_by_cropped_image(
    image: UploadFile = File(..., description="Cropped image file"),
    k: int = Query(10, ge=1, le=100, description="Number of results to return (top-K)"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Minimum similarity score threshold"),
    visual_weight: float = Query(0.6, ge=0.0, le=1.0, description="Weight for visual similarity"),
    color_weight: float = Query(0.4, ge=0.0, le=1.0, description="Weight for color similarity"),
    db: Session = Depends(get_db)
):
    """
    Search products using manually cropped image.
    This endpoint bypasses YOLO detection and searches directly with the provided cropped image.
    """
    # Read image bytes
    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return ProductSearchResponse(
            results=[],
            suggested_products=[],
            has_exact_match=False
        )

    # -------- Embedding infra --------
    model = get_model()
    client = get_qdrant_client()
    if model is None or client is None:
        raise HTTPException(500, "Embedding model or Qdrant not initialized")

    service = ProductService(db)
    exact_results = []
    all_suggested = []

    # Convert image to bytes
    try:
        _, img_encoded = cv2.imencode('.jpg', img)
        img_bytes = img_encoded.tobytes()
    except Exception as e:
        print(f"Error encoding image: {e}")
        return ProductSearchResponse(
            results=[],
            suggested_products=[],
            has_exact_match=False
        )

    # Search similar products
    try:
        hits = search_similar(
            model=model,
            client=client,
            image_bytes=img_bytes,
            top_k=k,
            visual_weight=visual_weight,
            color_weight=color_weight
        )
        
        # Filter by threshold
        filtered_hits = [hit for hit in hits if hit.get("score", 0.0) >= threshold]
        
        # Process exact matches
        if filtered_hits:
            best_hit = filtered_hits[0]
            product_id = best_hit.get("payload", {}).get("product_id")
            best_score = best_hit.get("score", 0.0)
            visual_score = best_hit.get("visual_score", 0.0)
            color_score = best_hit.get("color_score", 0.0)
            
            if product_id:
                try:
                    product = service.get_product(product_id)
                    exact_results.append(ProductSearchResult(
                        product=product,
                        score=best_score,
                        similarity_percent=best_score * 100.0,
                        bbox=None,  # No bbox for manual crop
                        visual_score=visual_score,
                        color_score=color_score
                    ))
                except HTTPException:
                    pass
        
        # Process suggestions
        seen_product_ids = {r.product.id for r in exact_results}
        for hit in hits[:k]:
            product_id = hit.get("payload", {}).get("product_id")
            score = hit.get("score", 0.0)
            visual_score = hit.get("visual_score", 0.0)
            color_score = hit.get("color_score", 0.0)
            
            if score >= threshold and product_id and product_id not in seen_product_ids:
                try:
                    product = service.get_product(product_id)
                    seen_product_ids.add(product_id)
                    all_suggested.append(ProductSearchResult(
                        product=product,
                        score=score,
                        similarity_percent=score * 100.0,
                        bbox=None,
                        visual_score=visual_score,
                        color_score=color_score
                    ))
                except HTTPException:
                    pass
    
    except Exception as e:
        print(f"Error searching cropped image: {str(e)}")
        return ProductSearchResponse(
            results=[],
            suggested_products=[],
            has_exact_match=False
        )

    # Sort by score descending
    exact_results.sort(key=lambda x: x.score, reverse=True)
    all_suggested.sort(key=lambda x: x.score, reverse=True)
    
    # Remove duplicates
    exact_product_ids = {r.product.id for r in exact_results}
    suggested_products = [r for r in all_suggested if r.product.id not in exact_product_ids]
    
    return ProductSearchResponse(
        results=exact_results,
        suggested_products=suggested_products[:k],
        has_exact_match=len(exact_results) > 0
    )


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

# # ------------------------------
# # Upload Product Image
# # ------------------------------
# @router.post(
#     "/{product_id}/upload-image/",
#     status_code=status.HTTP_201_CREATED,
#     summary="Upload product image"
# )
# def upload_product_image(
#     product_id: int = Path(..., description="The ID of the product to upload image for", example=1),
#     file: UploadFile = File(..., description="Image file to upload"),
#     is_primary: bool = Query(False, description="Set as primary image"),
#     db: Session = Depends(get_db)
# ):
#     # Validate file type
#     allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
#     file_extension = file.filename.split('.')[-1].lower()
    
#     if file_extension not in allowed_extensions:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
#         )
    
#     # Create upload directory
#     upload_dir = "uploads"
#     os.makedirs(upload_dir, exist_ok=True)
    
#     # Generate unique filename
#     file_name = f"{uuid.uuid4()}.{file_extension}"
#     file_path = os.path.join(upload_dir, file_name)
    
#     try:
#         # Save file synchronously
#         with open(file_path, "wb") as buffer:
#             buffer.write(file.file.read())
        
#         # Save file info to DB if needed
#         service = ProductService(db)
#         service.add_product_image(product_id, file_path, is_primary)
        
#         return {
#             "status": "success",
#             "message": "Image uploaded successfully",
#             "file_path": file_path,
#             "is_primary": is_primary
#         }
#     except Exception as e:
#         if os.path.exists(file_path):
#             os.remove(file_path)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error uploading file: {str(e)}"
#         )

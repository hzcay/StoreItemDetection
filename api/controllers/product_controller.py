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
from api.schemas.schema import ProductCreate, ProductResponse, ProductUpdate, ProductSearchResult,BoundingBox
from api.services.product_service import ProductService
from api.services import minio_client
from api.services.minio_client import put_object, MINIO_BUCKET
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
    response_model=List[ProductSearchResult],
    summary="Search products by image"
)
async def search_product_by_image(
    image: UploadFile = File(...),
    k: int = Query(20, ge=1, le=100),
    threshold: float = Query(0.0, ge=0.0, le=1.0),
    db: Session = Depends(get_db)
):
    # -------- Load image --------
    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    # -------- YOLO detection --------
    base_dir = os.path.dirname(__file__)
    yolo_path = os.path.join(base_dir, "..", "models", "best_new_15_12.pt")
    yolo_model = YOLO(yolo_path)
    detections = yolo_model(img)[0]

    # -------- Embedding infra --------
    model = get_model()
    client = get_qdrant_client()
    if model is None or client is None:
        raise HTTPException(500, "Embedding model or Qdrant not initialized")

    service = ProductService(db)
    seen = set()
    results = []

    draw_img = img.copy()  # copy for drawing

    for box in detections.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        ok, buf = cv2.imencode(".jpg", crop)
        if not ok:
            continue
        crop_bytes = buf.tobytes()

        # Search in Qdrant
        hits = search_similar(
            model=model,
            client=client,
            image_bytes=crop_bytes,
            top_k=k
        )
        hits = [h for h in hits if h["score"] >= threshold]
        if not hits:
            continue

        best = hits[0]
        product_id = best["payload"].get("product_id")
        if not product_id or product_id in seen:
            continue

        product = service.get_product(product_id)
        seen.add(product_id)

        score = best["score"]
        label = f"{product.name} ({score*100:.1f}%)"

        # =========================
        # DRAW BOUNDING BOX + LABEL
        # =========================
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, thickness)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_y = max(y1 - 10, th + 10)
        cv2.rectangle(draw_img, (x1, text_y - th - 6), (x1 + tw + 6, text_y), color, -1)
        cv2.putText(draw_img, label, (x1 + 3, text_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # =========================
        # ADD TO RESPONSE (with bbox)
        # =========================
        results.append(ProductSearchResult(
            product=product,
            score=score,
            similarity_percent=score * 100.0,
            bbox=BoundingBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2))
        ))

    # Optional: show image locally for debugging
    # cv2.imshow("Search Results", draw_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return results




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

from fastapi import HTTPException, status, UploadFile
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import uuid
from api.services.minio_client import minio_client, MINIO_BUCKET
from api.models.models import Product as ProductModel, ProductImage as ProductImageModel
from api.schemas.schema import ProductCreate, ProductResponse, ProductUpdate, ProductImageCreate, ProductImageResponse
from api.repositories.product_repository import ProductRepository
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
from api.services.minio_client import generate_presigned_url
class ProductService:
    def __init__(self, db: Session):
        self.db = db
        self.product_repo = ProductRepository(db)

    def handleSearchProductByImage(
    self,
    image: Optional[UploadFile]
) -> List[ProductResponse]:

        # Missing image
        if image is None:
            return []

        # Load YOLO model
        base_dir = os.path.dirname(__file__)  # folder of product_service.py
        model_path = os.path.join(base_dir, "..", "models", "best_new_15_12.pt")
        model = YOLO(model_path)


        # Read uploaded image bytes
        image_bytes = image.file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return []

        # Run detection
        results = model(img)
        detections = results[0]

        # Extract all crops
        crops = []
        for box in detections.boxes:
            # get xyxy bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # crop the region from the original image
            crop = img[y1:y2, x1:x2]

            # skip invalid crops
            if crop.size == 0:
                continue

            crops.append(crop)

        return []


    def create_product(self, product_data: ProductCreate) -> ProductResponse:
        """
        Create a new product with validation and business logic
        """
        try:
            # Convert Pydantic model to dict and create product
            product_dict = product_data.model_dump()
            db_product = self.product_repo.create_product(product_dict)
            return ProductResponse.model_validate(db_product.__dict__)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating product: {str(e)}"
            )

    def get_product(self, product_id: int) -> ProductResponse:
        """
        Get a product by ID with proper error handling and eager loading of images
        """
        # Get the product with eager loading of images
        db_product = self.product_repo.get_product_by_id(product_id)
        if not db_product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product with ID {product_id} not found"
            )
        
        # Convert images to proper format with presigned URLs
        images = []
        if hasattr(db_product, 'images') and db_product.images:
            images = [
                {
                    'id': img.id,
                    'image_url': generate_presigned_url(img.image_url),
                    'alt_text': img.alt_text,
                    'is_primary': img.is_primary,
                    'created_at': img.created_at
                }
                for img in db_product.images
            ]
        
        # Create response with all product data
        product_data = {
            **db_product.__dict__,
            'images': images
        }
        
        # Handle category if it's a relationship
        if hasattr(db_product, 'category') and db_product.category:
            product_data['category'] = db_product.category
        
        return ProductResponse.model_validate(product_data)

    def list_products(self, skip: int = 0, limit: int = 100) -> List[ProductResponse]:
        db_products = self.product_repo.get_products(skip=skip, limit=limit)
        results = []

        for product in db_products:
            images = []

            for img in product.images:
                images.append(ProductImageResponse(
                    id=img.id,
                    image_url=generate_presigned_url(img.image_url),
                    alt_text=img.alt_text,
                    is_primary=img.is_primary,
                    created_at=img.created_at
                ))

            results.append(ProductResponse(
                id=product.id,
                name=product.name,
                description=product.description,
                price=product.price,
                stock_quantity=product.stock_quantity,
                sku=product.sku,
                barcode=product.barcode,
                is_active=product.is_active,
                category=product.category,
                images=images,
                created_at=product.created_at,
                updated_at=product.updated_at
            ))

        return results


    def update_product(self, product_id: int, product_update: ProductUpdate) -> ProductResponse:
        """
        Update a product with validation
        """
        # Convert Pydantic model to dict and remove None values
        update_data = product_update.model_dump(exclude_unset=True)
        
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No update data provided"
            )
            
        db_product = self.product_repo.update_product(product_id, update_data)
        if not db_product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product with ID {product_id} not found"
            )
            
        return ProductResponse.model_validate(db_product.__dict__)

    def delete_product(self, product_id: int) -> bool:
        """
        Delete a product
        """
        success = self.product_repo.delete_product(product_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product with ID {product_id} not found"
            )
        return success

    def add_product_image(self, product_id: int, object_key: str, is_primary: bool = False) -> ProductImageResponse:
        """
        Add an image to a product
        """
        # Check if product exists
        product = self.product_repo.get_product_by_id(product_id)
        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product with ID {product_id} not found"
            )
        
        # If setting as primary, unset any existing primary image
        if is_primary:
            self.db.query(ProductImageModel).filter(
                ProductImageModel.product_id == product_id,
                ProductImageModel.is_primary == True
            ).update({ProductImageModel.is_primary: False})
            self.db.commit()
        
        # Add new image
        db_image = ProductImageModel(
            product_id=product_id,
            image_url=object_key,
            is_primary=is_primary
        )
        self.db.add(db_image)
        self.db.commit()
        self.db.refresh(db_image)
        return ProductImageResponse.model_validate(db_image.__dict__)

    def get_product_images(self, product_id: int) -> List[ProductImageResponse]:
        """
        Get all images for a product
        """
        product = self.product_repo.get_product_by_id(product_id)
        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product with ID {product_id} not found"
            )
        
        return [ProductImageResponse.model_validate(img.__dict__) for img in product.images]

    def delete_product_image(self, image_id: int) -> bool:
        """
        Delete a product image
        """
        db_image = self.db.query(ProductImageModel).filter(ProductImageModel.id == image_id).first()
        if not db_image:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image with ID {image_id} not found"
            )
        
        minio_client.remove_object(
            MINIO_BUCKET,
            db_image.image_url
        )
        
        # Delete the database record
        self.db.delete(db_image)
        self.db.commit()
        return True
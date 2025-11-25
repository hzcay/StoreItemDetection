from fastapi import HTTPException, status, UploadFile
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import uuid

from models import Product as ProductModel, ProductImage as ProductImageModel
from schemas.schema import ProductCreate, ProductResponse, ProductUpdate, ProductImageCreate, ProductImageResponse
from repositories.product_repository import ProductRepository

class ProductService:
    def __init__(self, db: Session):
        self.db = db
        self.product_repo = ProductRepository(db)

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
        Get a product by ID with proper error handling
        """
        db_product = self.product_repo.get_product_by_id(product_id)
        if not db_product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product with ID {product_id} not found"
            )
        return ProductResponse.model_validate(db_product.__dict__)

    def list_products(self, skip: int = 0, limit: int = 100) -> List[ProductResponse]:
        """
        List products with pagination
        """
        db_products = self.product_repo.get_products(skip=skip, limit=limit)
        return [ProductResponse.model_validate(p.__dict__) for p in db_products]

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

    def add_product_image(self, product_id: int, image_url: str, is_primary: bool = False) -> ProductImageResponse:
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
            image_url=image_url,
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
        
        # Delete the image file
        try:
            if os.path.exists(db_image.image_url):
                os.remove(db_image.image_url)
        except Exception as e:
            print(f"Error deleting image file: {str(e)}")
        
        # Delete the database record
        self.db.delete(db_image)
        self.db.commit()
        return True
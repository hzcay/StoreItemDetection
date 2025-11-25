from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from models import Product as ProductModel, ProductImage as ProductImageModel
from schemas.schema import ProductCreate, ProductImageCreate

class ProductRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_product(self, product_data: dict):
        """Create a new product"""
        db_product = ProductModel(**product_data)
        self.db.add(db_product)
        self.db.commit()
        self.db.refresh(db_product)
        return db_product

    def get_product_by_id(self, product_id: int) -> Optional[ProductModel]:
        """Get a product by ID"""
        return self.db.query(ProductModel).filter(ProductModel.id == product_id).first()

    def get_product_by_sku(self, sku: str) -> Optional[ProductModel]:
        """Get a product by SKU"""
        return self.db.query(ProductModel).filter(ProductModel.sku == sku).first()

    def get_products(self, skip: int = 0, limit: int = 100) -> List[ProductModel]:
        """Get list of products with pagination"""
        return self.db.query(ProductModel).offset(skip).limit(limit).all()

    def update_product(self, product_id: int, product_update: dict) -> Optional[ProductModel]:
        """Update a product"""
        db_product = self.get_product_by_id(product_id)
        if not db_product:
            return None
            
        for field, value in product_update.items():
            setattr(db_product, field, value)
            
        db_product.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(db_product)
        return db_product

    def delete_product(self, product_id: int) -> bool:
        """Delete a product and its images"""
        db_product = self.get_product_by_id(product_id)
        if not db_product:
            return False
            
        # Delete associated images
        self.db.query(ProductImageModel).filter(ProductImageModel.product_id == product_id).delete()
        
        # Delete the product
        self.db.delete(db_product)
        self.db.commit()
        return True
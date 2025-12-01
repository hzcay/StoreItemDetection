from fastapi import HTTPException, status, UploadFile
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import uuid

from models import Product as ProductModel, ProductImage as ProductImageModel
from schemas.schema import CategoryCreate, CategoryResponse
from repositories.categories_repository import CategoriesRepository

class CategoriesService:
    def __init__(self, db: Session):
        self.db = db
        self.categories_repo = CategoriesRepository(db)

    def create_category(self, category_data: CategoryCreate) -> CategoryResponse:
        """Create a new category"""
        try:
            category_dict = category_data.model_dump()
            db_product = self.categories_repo.create_category(category_dict)
            return CategoryResponse.model_validate(db_product.__dict__)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating category: {str(e)}"
            )

    def get_category_by_id(self, category_id: int) -> CategoryResponse:
        """Get a category by ID"""
        try:
            db_category = self.categories_repo.get_category_by_id(category_id)
            if not db_category:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Category with ID {category_id} not found"
                )
            return CategoryResponse.model_validate(db_category.__dict__)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting category: {str(e)}"
            )
    
    def get_categories(self, skip: int = 0, limit: int = 100) -> List[CategoryResponse]:
        """Get all categories"""
        try:
            db_categories = self.categories_repo.get_categories(skip=skip, limit=limit)
            return [CategoryResponse.model_validate(c.__dict__) for c in db_categories]
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting categories: {str(e)}"
            )
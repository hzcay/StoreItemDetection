from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from api.models.models import Category as CategoryModel
from api.schemas.schema import CategoryCreate

class CategoriesRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_category(self, category_data: dict):
        """Create a new category"""
        db_category = CategoryModel(**category_data)
        self.db.add(db_category)
        self.db.commit()
        self.db.refresh(db_category)
        return db_category

    def get_category_by_id(self, category_id: int) -> Optional[CategoryModel]:
        """Get a category by ID"""
        return self.db.query(CategoryModel).filter(CategoryModel.id == category_id).first()

    def get_categories(self, skip: int = 0, limit: int = 100) -> List[CategoryModel]:
        """Get all categories"""
        return self.db.query(CategoryModel).offset(skip).limit(limit).all()
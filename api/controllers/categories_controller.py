from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query, Path, Form
from typing import List
from sqlalchemy.orm import Session
import os
import uuid

from database import get_db
from schemas.schema import CategoryCreate, CategoryResponse
from services.categories_service import CategoriesService

router = APIRouter(
    prefix="/api/categories",
    tags=["Categories"],
    responses={
        404: {"description": "Category not found"},
        500: {"description": "Internal server error"}
    }
)

@router.post(
    "/",
    response_model=CategoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new category"
)
def create_category(
    name: str = Form(...),
    description: str = Form(None),
    db: Session = Depends(get_db)
):
    service = CategoriesService(db)
    category = service.create_category(CategoryCreate(name=name, description=description))
    return category


@router.get(
    "/{category_id}",
    response_model=CategoryResponse,
    summary="Get category by ID"
)
def get_category(
    category_id: int = Path(..., description="The ID of the category to retrieve", example=1),
    db: Session = Depends(get_db)
):
    service = CategoriesService(db)
    category = service.get_category_by_id(category_id)
    return category


@router.get(
    "/",
    response_model=List[CategoryResponse],
    summary="List all categories"
)
def list_categories(
    skip: int = Query(0, description="Number of records to skip", ge=0),
    limit: int = Query(100, description="Max number of records to return", le=1000),
    db: Session = Depends(get_db)
):
    service = CategoriesService(db)
    categories = service.get_categories(skip=skip, limit=limit)
    return categories
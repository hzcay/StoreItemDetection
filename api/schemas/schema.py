from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl

# Base schemas
class CategoryBase(BaseModel):
    name: str = Field(..., max_length=100, description="Name of the category")
    description: Optional[str] = Field(None, description="Description of the category")

class CategoryCreate(CategoryBase):
    pass

class CategoryResponse(CategoryBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# ProductImage schemas
class ProductImageBase(BaseModel):
    image_url: str = Field(..., description="URL or file path of the product image")
    alt_text: Optional[str] = Field(None, max_length=200, description="Alternative text for the image")
    is_primary: bool = Field(False, description="Whether this is the primary image for the product")

class ProductImageCreate(ProductImageBase):
    pass

class ProductImageResponse(ProductImageBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Product schemas
class ProductBase(BaseModel):
    name: str = Field(..., max_length=200, description="Name of the product")
    description: Optional[str] = Field(None, description="Detailed description of the product")
    price: float = Field(..., gt=0, description="Price of the product (must be greater than 0)")
    stock_quantity: int = Field(0, ge=0, description="Available quantity in stock")
    sku: Optional[str] = Field(None, max_length=100, description="Stock Keeping Unit")
    barcode: Optional[str] = Field(None, max_length=100, description="Product barcode")
    is_active: bool = Field(True, description="Whether the product is active")
    category_id: Optional[int] = Field(None, description="ID of the product's category")

class ProductCreate(ProductBase):
    pass

class ProductResponse(ProductBase):
    id: int
    category: Optional[CategoryResponse] = None
    images: List[ProductImageResponse] = []
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# For updating products (all fields optional)
class ProductUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = None
    price: Optional[float] = Field(None, gt=0)
    stock_quantity: Optional[int] = Field(None, ge=0)
    sku: Optional[str] = Field(None, max_length=100)
    barcode: Optional[str] = Field(None, max_length=100)
    is_active: Optional[bool] = None
    category_id: Optional[int] = None
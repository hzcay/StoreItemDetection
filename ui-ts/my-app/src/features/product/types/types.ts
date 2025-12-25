// src/features/product/types/types.ts

// Base types
export interface Category {
  id: number;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
}

export interface ProductImage {
  id: number;
  image_url: string;
  is_primary: boolean;
  created_at: string;
  updated_at: string;
}

export interface Product {
  id: number;
  name: string;
  description: string | null;
  price: number;
  stock_quantity: number;
  sku: string;
  barcode: string;
  is_active: boolean;
  category_id: number | null;
  category: Category | null;
  images: ProductImage[];
  created_at: string;
  updated_at: string;
}

// Request types
export interface CreateProductDto {
  name: string;
  description?: string;
  price: number;
  stock_quantity?: number;
  sku?: string;
  barcode?: string;
  is_active?: boolean;
  category_id?: number | null;
  images?: File[];
}

export interface UpdateProductDto {
  name?: string;
  description?: string | null;
  price?: number;
  stock_quantity?: number;
  sku?: string;
  barcode?: string;
  is_active?: boolean;
  category_id?: number | null;
}

// Response types
export interface ProductResponse extends Omit<Product, 'images'> {
  images: ProductImage[];
}

export interface SearchByImageResult {
  product: Product;
  score: number;
  similarity_percent: number;
}

// Query params
export interface ListProductsParams {
  skip?: number;
  limit?: number;
}

export interface SearchByImageParams {
  image: File;
  k?: number;
  threshold?: number;
}
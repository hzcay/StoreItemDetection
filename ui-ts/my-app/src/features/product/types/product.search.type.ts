// types/product.search.type.ts

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Category {
  id: number;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
}

export interface ProductImage {
  id: number;
  image_url: string;
  alt_text: string | null;
  is_primary: boolean;
  created_at: string;
}

export interface Product {
  id: number;
  name: string;
  description: string;
  price: number;
  stock_quantity: number;
  sku: string;
  barcode: string;
  is_active: boolean;
  category_id: number;
  category?: Category;
  images: ProductImage[];
  created_at: string;
  updated_at: string;
}

export interface SearchResult {
  product: Product;
  score: number;
  similarity_percent: number;
  bbox: BoundingBox | null;  // Added bbox field
  visual_score: number;
  color_score: number;
}

export interface SearchResponse {
  results: SearchResult[];  // Exact matches (score >= threshold)
  suggested_products: SearchResult[];  // Suggested products (top-K, may include < threshold)
  has_exact_match: boolean;
}
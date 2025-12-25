// src/features/product/services/product.api.ts
import {
    Product,
    CreateProductDto,
    UpdateProductDto,
    ListProductsParams,
    SearchByImageResult
} from '../types/types';

const API_BASE_URL = 'http://localhost:8000/api/products';

async function handleResponse<T>(response: Response): Promise<T> {
    const contentType = response.headers.get('content-type');

    if (!response.ok) {
        if (contentType?.includes('application/json')) {
            const error = await response.json();
            throw new Error(error.detail || 'An error occurred');
        } else {
            const text = await response.text();
            throw new Error(text || 'An error occurred');
        }
    }

    if (response.status === 204) { // No Content
        return undefined as unknown as T;
    }

    return response.json() as Promise<T>;
}

export const productApi = {
    // Create a new product
    async create(productData: CreateProductDto): Promise<Product> {
        const formData = new FormData();

        // Append all fields to form data
        Object.entries(productData).forEach(([key, value]) => {
            if (key === 'images' && value) {
                (value as File[]).forEach((file, index) => {
                    formData.append('images', file);
                });
            } else if (value !== undefined && value !== null) {
                formData.append(key, value.toString());
            }
        });

        const response = await fetch(API_BASE_URL, {
            method: 'POST',
            body: formData,
        });

        return handleResponse<Product>(response);
    },

    // Get all products with pagination
    async list(params: ListProductsParams = {}): Promise<Product[]> {
        const { skip = 0, limit = 100 } = params;
        const queryParams = new URLSearchParams({
            skip: skip.toString(),
            limit: limit.toString(),
        });

        const url = `${API_BASE_URL}/?${queryParams}`;
        console.log('Fetching products from:', url);

        try {
            const response = await fetch(url);
            console.log('Response status:', response.status, response.statusText);

            // Clone the response to read it without consuming the stream
            const responseClone = response.clone();
            const data = await response.json().catch(e => {
                console.error('Error parsing JSON:', e);
                return null;
            });

            console.log('Response data:', data);

            // Return the original response to be handled by handleResponse
            return handleResponse<Product[]>(responseClone);
        } catch (error) {
            console.error('Error in list API call:', error);
            throw error;
        }
    },

    // Get product by ID
    async getById(id: number): Promise<Product> {
        const response = await fetch(`${API_BASE_URL}/${id}`);
        return handleResponse<Product>(response);
    },

    // Update product
    async update(id: number, data: UpdateProductDto): Promise<Product> {
        const response = await fetch(`${API_BASE_URL}/${id}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        return handleResponse<Product>(response);
    },

    // Delete product
    async delete(id: number): Promise<void> {
        const response = await fetch(`${API_BASE_URL}/${id}`, {
            method: 'DELETE',
        });

        await handleResponse<void>(response);
    },

    // Search products by image
    async searchByImage(
        image: File,
        k: number = 20,
        threshold: number = 0.0
    ): Promise<SearchByImageResult[]> {
        const formData = new FormData();
        formData.append('image', image);

        const queryParams = new URLSearchParams({
            k: k.toString(),
            threshold: threshold.toString(),
        });

        const response = await fetch(`${API_BASE_URL}/search-by-image?${queryParams}`, {
            method: 'POST',
            body: formData,
        });

        return handleResponse<SearchByImageResult[]>(response);
    },
};

export default productApi;
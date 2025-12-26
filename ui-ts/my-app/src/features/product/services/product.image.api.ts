// src/features/product/services/product.image.api.ts
import { SearchResult } from '../types/product.search.type';

const API_BASE_URL = 'http://localhost:8000/api';

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

    if (response.status === 204) {
        return undefined as unknown as T;
    }

    return response.json() as Promise<T>;
}

export const searchProductsByImage = async (image: File): Promise<SearchResult[]> => {
    const formData = new FormData();
    formData.append('image', image);

    try {
        const response = await fetch(`${API_BASE_URL}/products/search-by-image`, {
            method: 'POST',
            body: formData,
            // Don't set Content-Type header, let the browser set it with the correct boundary
            headers: {
                'Accept': 'application/json',
            },
            credentials: 'include', // Include cookies for authentication if needed
        });

        return handleResponse<SearchResult[]>(response);
    } catch (error) {
        console.error('Error searching products by image:', error);
        throw error;
    }
};
// src/features/product/services/product.image.api.ts
import { SearchResponse } from '../types/product.search.type';

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

export const searchProductsByImage = async (
    image: File,
    k: number = 10,
    threshold: number = 0.5,
    visualWeight: number = 0.6,
    colorWeight: number = 0.4,
    useYolo: boolean = true
): Promise<SearchResponse> => {
    const formData = new FormData();
    formData.append('image', image);

    const params = new URLSearchParams({
        k: k.toString(),
        threshold: threshold.toString(),
        visual_weight: visualWeight.toString(),
        color_weight: colorWeight.toString(),
        use_yolo: useYolo.toString(),
    });

    try {
        const response = await fetch(`${API_BASE_URL}/products/search-by-image?${params}`, {
            method: 'POST',
            body: formData,
            // Don't set Content-Type header, let the browser set it with the correct boundary
            headers: {
                'Accept': 'application/json',
            },
            credentials: 'include', // Include cookies for authentication if needed
        });

        return handleResponse<SearchResponse>(response);
    } catch (error) {
        console.error('Error searching products by image:', error);
        throw error;
    }
};

export const searchProductsByCroppedImage = async (
    image: File,
    k: number = 10,
    threshold: number = 0.5,
    visualWeight: number = 0.6,
    colorWeight: number = 0.4
): Promise<SearchResponse> => {
    const formData = new FormData();
    formData.append('image', image);

    const params = new URLSearchParams({
        k: k.toString(),
        threshold: threshold.toString(),
        visual_weight: visualWeight.toString(),
        color_weight: colorWeight.toString(),
    });

    try {
        const response = await fetch(`${API_BASE_URL}/products/search-by-cropped-image?${params}`, {
            method: 'POST',
            body: formData,
            headers: {
                'Accept': 'application/json',
            },
            credentials: 'include',
        });

        return handleResponse<SearchResponse>(response);
    } catch (error) {
        console.error('Error searching products by cropped image:', error);
        throw error;
    }
};
// services/category.api.ts
import { Category, CreateCategoryDto, ListCategoriesParams, UpdateCategoryDto } from '../types/types';

const API_BASE_URL = 'http://localhost:8000/api/categories';

async function handleResponse<T>(response: Response): Promise<T> {
    const contentType = response.headers.get('content-type');

    // Handle non-JSON responses (like HTML error pages)
    if (!contentType || !contentType.includes('application/json')) {
        const text = await response.text();
        throw new Error(`Invalid response format. Expected JSON but got: ${text.substring(0, 100)}...`);
    }

    const data = await response.json();
    if (!response.ok) {
        const error = data as { message?: string; detail?: string | { message: string } };
        const errorMessage = error?.message ||
            (typeof error?.detail === 'string' ? error.detail : error.detail?.message) ||
            `HTTP error! status: ${response.status}`;
        throw new Error(errorMessage);
    }
    return data as T;
}

export const categoryApi = {
    // Create a new category
    async create(categoryData: CreateCategoryDto): Promise<Category> {
        const formData = new FormData();
        formData.append('name', categoryData.name);
        if (categoryData.description) {
            formData.append('description', categoryData.description);
        }

        const response = await fetch(API_BASE_URL, {
            method: 'POST',
            body: formData,
        });

        return handleResponse<Category>(response);
    },

    // Get a single category by ID
    async getById(id: number): Promise<Category> {
        const response = await fetch(`${API_BASE_URL}/${id}`);
        return handleResponse<Category>(response);
    },

    // List all categories with pagination
    async list(params: ListCategoriesParams = {}): Promise<Category[]> {
        const { skip = 0, limit = 100 } = params;
        const queryParams = new URLSearchParams({
            skip: skip.toString(),
            limit: limit.toString(),
        });

        const response = await fetch(`${API_BASE_URL}/?${queryParams}`);
        return handleResponse<Category[]>(response);
    },

    // Update a category
    async update(id: number, data: UpdateCategoryDto): Promise<Category> {
        const formData = new FormData();
        if (data.name) formData.append('name', data.name);
        if (data.description !== undefined) {
            formData.append('description', data.description || '');
        }

        const response = await fetch(`${API_BASE_URL}/${id}`, {
            method: 'PUT',
            body: formData,
        });

        return handleResponse<Category>(response);
    },

    // Delete a category
    async delete(id: number): Promise<void> {
        const response = await fetch(`${API_BASE_URL}/${id}`, {
            method: 'DELETE',
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.message || 'Failed to delete category');
        }
    },
};

export default categoryApi;
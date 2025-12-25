// types/types.ts
export interface Category {
    id: number;
    name: string;
    description?: string;
    created_at: string;
    updated_at: string;
}

export interface CreateCategoryDto {
    name: string;
    description?: string;
}

export interface UpdateCategoryDto {
    name?: string;
    description?: string | null;
}

export interface ListCategoriesParams {
    skip?: number;
    limit?: number;
}

export interface ErrorResponse {
    message: string;
    statusCode?: number;
    error?: string;
}
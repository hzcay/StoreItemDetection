// hooks/useCategories.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { categoryApi } from '../services/category.api';
import { Category, CreateCategoryDto, UpdateCategoryDto } from '../types/types';

export const useCategories = (params = { skip: 0, limit: 100 }) => {
    return useQuery<Category[], Error>({
        queryKey: ['categories', params],
        queryFn: () => categoryApi.list(params),
    });
};

export const useCategory = (id: number) => {
    return useQuery<Category, Error>({
        queryKey: ['category', id],
        queryFn: () => categoryApi.getById(id),
        enabled: !!id,
    });
};

export const useCreateCategory = () => {
    const queryClient = useQueryClient();

    return useMutation<Category, Error, CreateCategoryDto>({
        mutationFn: categoryApi.create,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['categories'] });
        },
    });
};

export const useUpdateCategory = () => {
    const queryClient = useQueryClient();

    return useMutation<Category, Error, { id: number; data: UpdateCategoryDto }>({
        mutationFn: ({ id, data }) => categoryApi.update(id, data),
        onSuccess: (_, variables) => {
            queryClient.invalidateQueries({ queryKey: ['categories'] });
            queryClient.invalidateQueries({ queryKey: ['category', variables.id] });
        },
    });
};

export const useDeleteCategory = () => {
    const queryClient = useQueryClient();

    return useMutation<void, Error, number>({
        mutationFn: (id: number) => categoryApi.delete(id),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['categories'] });
        },
    });
};
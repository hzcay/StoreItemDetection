import React, { useState, ChangeEvent, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search, Image as ImageIcon, X } from "lucide-react";

interface ProductSearchProps {
    onTextSearch?: (text: string) => void;
    onImageUpload?: (file: File | null) => void;
}

export const ProductSearch: React.FC<ProductSearchProps> = ({
    onTextSearch,
    onImageUpload,
}) => {
    const [searchTerm, setSearchTerm] = useState("");
    const [imagePreview, setImagePreview] = useState<string | null>(null);

    // 🔹 Debounce text search
    useEffect(() => {
        const timeout = setTimeout(() => {
            onTextSearch?.(searchTerm);
        }, 300);

        return () => clearTimeout(timeout);
    }, [searchTerm, onTextSearch]);

    // 🔹 Cleanup object URL
    useEffect(() => {
        return () => {
            if (imagePreview) URL.revokeObjectURL(imagePreview);
        };
    }, [imagePreview]);

    const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files?.[0]) return;

        const file = e.target.files[0];
        const url = URL.createObjectURL(file);

        setImagePreview(url);
        onImageUpload?.(file);

        e.target.value = "";
    };

    const clearImage = () => {
        setImagePreview(null);
        onImageUpload?.(null);
    };

    return (
        <div className="relative w-full md:w-[520px]">
            {/* 🔍 Search Icon */}
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />

            {/* 🖼 Image preview inside bar */}
            {imagePreview && (
                <div className="absolute right-16 top-1/2 -translate-y-1/2 flex items-center gap-1">
                    <img
                        src={imagePreview}
                        alt="Preview"
                        className="h-8 w-8 rounded object-cover border"
                    />
                    <button
                        onClick={clearImage}
                        className="text-muted-foreground hover:text-destructive"
                    >
                        <X className="h-4 w-4" />
                    </button>
                </div>
            )}

            {/* 🖊 Input */}
            <Input
                type="text"
                placeholder="Search products or upload an image..."
                className={`pl-10 pr-24`}
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
            />

            {/* 🖼 Image Upload Button */}
            <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
                id="image-upload"
            />
            <label htmlFor="image-upload">
                <Button
                    variant="ghost"
                    size="icon"
                    className="absolute right-2 top-1/2 -translate-y-1/2"
                >
                    <ImageIcon className="h-5 w-5" />
                </Button>
            </label>
        </div>
    );
};

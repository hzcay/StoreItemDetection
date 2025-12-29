import React, { useState, ChangeEvent, useEffect, useRef } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search, Image as ImageIcon, X, Camera } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { CameraCapture } from "./CameraCapture";
import { ImageCropModal } from "./ImageCropModal";

interface ProductSearchProps {
    onTextSearch?: (text: string) => void;
    onImageUpload?: (file: File) => void;
}

export const ProductSearch: React.FC<ProductSearchProps> = ({
    onTextSearch,
    onImageUpload,
}) => {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [searchTerm, setSearchTerm] = useState("");
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [showCameraModal, setShowCameraModal] = useState(false);
    const [showCropModal, setShowCropModal] = useState(false);
    const [capturedImage, setCapturedImage] = useState<string | null>(null);
    const navigate = useNavigate();

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
            if (capturedImage) URL.revokeObjectURL(capturedImage);
        };
    }, [imagePreview, capturedImage]);

    const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files?.[0]) return;

        const file = e.target.files[0];
        const url = URL.createObjectURL(file);

        setImagePreview(url);

        // Call the onImageUpload prop if it exists
        if (onImageUpload) {
            onImageUpload(file);
        } else {
            // Fallback to the original behavior if onImageUpload is not provided
            const reader = new FileReader();
            reader.onloadend = () => {
                const base64String = reader.result as string;
                sessionStorage.setItem('uploadedImage', base64String);
                navigate('/products-image-search');
            };
            reader.readAsDataURL(file);
        }

        e.target.value = "";
    };

    const clearImage = () => {
        setImagePreview(null);
        sessionStorage.removeItem('uploadedImage');
    };

    const handleCameraCapture = (imageFile: File) => {
        const imageUrl = URL.createObjectURL(imageFile);
        setCapturedImage(imageUrl);
        setImagePreview(imageUrl);
        // Tự động mở crop modal sau khi chụp ảnh
        setShowCropModal(true);
    };

    const handleCropComplete = (croppedImage: File) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64String = reader.result as string;
            sessionStorage.setItem('uploadedImage', base64String);
            setShowCropModal(false);
            navigate('/products-image-search');
        };
        reader.readAsDataURL(croppedImage);
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
                ref={fileInputRef}
            />
            <div className="absolute right-2 top-1/2 -translate-y-1/2 flex gap-1">
                <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setShowCameraModal(true)}
                    title="Chụp ảnh"
                >
                    <Camera className="h-5 w-5" />
                </Button>
                <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => fileInputRef.current?.click()}
                    title="Tải ảnh lên"
                >
                    <ImageIcon className="h-5 w-5" />
                </Button>
            </div>

            {/* Camera Modal */}
            <CameraCapture
                isOpen={showCameraModal}
                onClose={() => setShowCameraModal(false)}
                onCapture={handleCameraCapture}
            />

            {/* Crop Modal */}
            {capturedImage && (
                <ImageCropModal
                    imageSrc={capturedImage}
                    isOpen={showCropModal}
                    onClose={() => {
                        setShowCropModal(false);
                        setCapturedImage(null);
                    }}
                    onCropComplete={handleCropComplete}
                />
            )}
        </div>
    );
};
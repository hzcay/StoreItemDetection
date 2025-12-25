import { useRef, useState } from "react";
import Cropper from "react-cropper";
import "cropperjs/dist/cropper.min.css";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Loader2, X } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { toast } from "sonner";
import { CreateProductDto } from "../types/types";

interface ProductFormProps {
    formData: CreateProductDto;
    setFormData: (data: CreateProductDto) => void;
    categories: Array<{ id: number; name: string }>;
    onSubmit: (e: React.FormEvent) => void;
    isSubmitting: boolean;
}

export default function ProductForm({
    formData,
    setFormData,
    categories,
    onSubmit,
    isSubmitting
}: ProductFormProps) {
    const cropperRef = useRef<HTMLImageElement & { cropper?: Cropper }>(null);
    const [cropImage, setCropImage] = useState<string | null>(null);
    const [previewImages, setPreviewImages] = useState<string[]>([]);

    const onSelectFile = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files?.length) return;
        setCropImage(URL.createObjectURL(e.target.files[0]));
    };

    const saveCroppedImage = async () => {
        const cropper = cropperRef.current?.cropper;
        if (!cropper) return;

        try {
            const canvas = cropper.getCroppedCanvas({
                width: 800,
                height: 800,
                imageSmoothingQuality: "high",
            });

            const blob = await new Promise<Blob | null>((resolve) =>
                canvas.toBlob(resolve, "image/jpeg", 0.9)
            );

            if (!blob) throw new Error("Crop failed");

            const file = new File([blob], `product-${Date.now()}.jpg`, {
                type: "image/jpeg",
            });

            setFormData({
                ...formData,
                images: [...(formData.images || []), file],
            });

            setPreviewImages((prev) => [...prev, URL.createObjectURL(file)]);
            setCropImage(null);
        } catch (error) {
            console.error(error);
            toast.error("Failed to crop image");
        }
    };

    const handleFormChange = (field: keyof CreateProductDto, value: any) => {
        setFormData({ ...formData, [field]: value });
    };

    return (
        <>
            <form onSubmit={onSubmit} className="space-y-4 bg-white p-6 rounded-lg">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Name */}
                    <div className="space-y-2">
                        <Label>Product Name *</Label>
                        <Input
                            value={formData.name}
                            onChange={(e) => handleFormChange("name", e.target.value)}
                            required
                        />
                    </div>

                    {/* Category */}
                    <div className="space-y-2">
                        <Label>Category</Label>
                        <Select
                            value={formData.category_id?.toString() || ""}
                            onValueChange={(v) => handleFormChange("category_id", Number(v))}
                        >
                            <SelectTrigger className="bg-white">
                                <SelectValue placeholder="Select category" />
                            </SelectTrigger>
                            <SelectContent className="bg-white">
                                {categories.map((c) => (
                                    <SelectItem key={c.id} value={c.id.toString()}>
                                        {c.name}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>

                    {/* Price */}
                    <div className="space-y-2">
                        <Label>Price *</Label>
                        <Input
                            type="number"
                            step="0.01"
                            value={formData.price || ""}
                            onChange={(e) =>
                                handleFormChange("price", parseFloat(e.target.value))
                            }
                            required
                        />
                    </div>

                    {/* Stock */}
                    <div className="space-y-2">
                        <Label>Stock Quantity</Label>
                        <Input
                            type="number"
                            value={formData.stock_quantity || ""}
                            onChange={(e) =>
                                handleFormChange("stock_quantity", Number(e.target.value))
                            }
                        />
                    </div>

                    {/* Active */}
                    <div className="flex items-center gap-2">
                        <Switch
                            checked={formData.is_active ?? true}
                            onCheckedChange={(v) => handleFormChange("is_active", v)}
                        />
                        <Label>Active</Label>
                    </div>
                </div>

                {/* Description */}
                <div className="space-y-2">
                    <Label>Description</Label>
                    <textarea
                        className="w-full rounded-md border p-2"
                        value={formData.description || ""}
                        onChange={(e) =>
                            handleFormChange("description", e.target.value)
                        }
                    />
                </div>

                {/* Image Upload */}
                <div className="space-y-2">
                    <Label>Product Images</Label>
                    <Input type="file" accept="image/*" onChange={onSelectFile} />

                    <div className="flex gap-4 mt-2">
                        {previewImages.map((img, i) => (
                            <div key={i} className="relative">
                                <img
                                    src={img}
                                    className="h-24 w-24 rounded-md object-cover"
                                />
                                <button
                                    type="button"
                                    className="absolute -top-2 -right-2 bg-red-500 text-white p-1 rounded-full"
                                    onClick={() => {
                                        setPreviewImages(prev =>
                                            prev.filter((_, idx) => idx !== i)
                                        );
                                        handleFormChange(
                                            "images",
                                            (formData.images || []).filter((_, idx) => idx !== i)
                                        );
                                    }}
                                >
                                    <X size={14} />
                                </button>
                            </div>
                        ))}
                    </div>
                </div>

                <DialogFooter>
                    <Button
                        type="submit"
                        disabled={isSubmitting}
                        className="bg-blue-600 hover:bg-blue-700 text-white"
                    >
                        {isSubmitting && (
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        )}
                        Save Product
                    </Button>
                </DialogFooter>
            </form>

            {/* Cropper Modal */}
            {cropImage && (
                <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center">
                    <div className="bg-background p-6 rounded-lg w-full max-w-2xl">
                        <DialogHeader>
                            <DialogTitle>Crop Image</DialogTitle>
                        </DialogHeader>

                        <Cropper
                            ref={cropperRef}
                            src={cropImage}
                            aspectRatio={NaN}  // This allows free resizing
                            viewMode={1}
                            guides={true}
                            className="h-96 w-full"
                            minCropBoxWidth={100}
                            minCropBoxHeight={100}
                        />

                        <div className="mt-4 flex justify-end gap-2">
                            <Button
                                variant="outline"
                                onClick={() => setCropImage(null)}
                            >
                                Cancel
                            </Button>
                            <Button
                                onClick={saveCroppedImage}
                                className="bg-black hover:bg-gray-800 text-white"
                            >
                                Save Image
                            </Button>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}

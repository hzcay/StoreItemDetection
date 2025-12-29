import { useState, useRef } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import Cropper from "react-cropper";
import "cropperjs/dist/cropper.min.css";

interface ImageCropModalProps {
    imageSrc: string;
    isOpen: boolean;
    onClose: () => void;
    onCropComplete: (croppedImage: File) => void;
}

export function ImageCropModal({ imageSrc, isOpen, onClose, onCropComplete }: ImageCropModalProps) {
    const cropperRef = useRef<HTMLImageElement & { cropper?: Cropper }>(null);
    const [isProcessing, setIsProcessing] = useState(false);

    const handleSave = async () => {
        const cropper = cropperRef.current?.cropper;
        if (!cropper) return;

        setIsProcessing(true);
        try {
            const canvas = cropper.getCroppedCanvas({
                imageSmoothingQuality: "high",
            });

            const blob = await new Promise<Blob | null>((resolve) =>
                canvas.toBlob(resolve, "image/jpeg", 0.9)
            );

            if (!blob) throw new Error("Crop failed");

            const file = new File([blob], `cropped-${Date.now()}.jpg`, {
                type: "image/jpeg",
            });

            onCropComplete(file);
            onClose();
        } catch (error) {
            console.error("Error cropping image:", error);
        } finally {
            setIsProcessing(false);
        }
    };

    return (
        <Dialog open={isOpen} onOpenChange={onClose}>
            <DialogContent className="max-w-4xl max-h-[90vh] overflow-auto">
                <DialogHeader>
                    <DialogTitle>Cắt ảnh để tìm kiếm</DialogTitle>
                </DialogHeader>
                <div className="w-full h-[500px]">
                    <Cropper
                        ref={cropperRef}
                        src={imageSrc}
                        style={{ height: "100%", width: "100%" }}
                        aspectRatio={undefined}
                        guides={true}
                        minCropBoxWidth={50}
                        minCropBoxHeight={50}
                        background={true}
                        responsive={true}
                        autoCropArea={0.8}
                        checkOrientation={false}
                    />
                </div>
                <DialogFooter>
                    <Button variant="outline" onClick={onClose} disabled={isProcessing}>
                        Hủy
                    </Button>
                    <Button onClick={handleSave} disabled={isProcessing}>
                        {isProcessing ? "Đang xử lý..." : "Tìm kiếm"}
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}


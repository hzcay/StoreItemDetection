import { useState, useRef, useEffect } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Camera, X } from "lucide-react";

interface CameraCaptureProps {
    isOpen: boolean;
    onClose: () => void;
    onCapture: (imageFile: File) => void;
}

export function CameraCapture({ isOpen, onClose, onCapture }: CameraCaptureProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const [isCapturing, setIsCapturing] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (isOpen) {
            startCamera();
        } else {
            stopCamera();
        }

        return () => {
            stopCamera();
        };
    }, [isOpen]);

    const startCamera = async () => {
        try {
            setError(null);
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: "environment", // Use back camera on mobile
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                },
            });

            streamRef.current = stream;
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
            }
        } catch (err) {
            console.error("Error accessing camera:", err);
            setError("Không thể truy cập camera. Vui lòng kiểm tra quyền truy cập.");
        }
    };

    const stopCamera = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach((track) => track.stop());
            streamRef.current = null;
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
    };

    const capturePhoto = () => {
        const video = videoRef.current;
        if (!video) return;

        setIsCapturing(true);
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext("2d");
        if (!ctx) {
            setIsCapturing(false);
            return;
        }

        ctx.drawImage(video, 0, 0);

        canvas.toBlob((blob) => {
            if (!blob) {
                setIsCapturing(false);
                return;
            }

            const file = new File([blob], `camera-${Date.now()}.jpg`, {
                type: "image/jpeg",
            });

            onCapture(file);
            setIsCapturing(false);
            stopCamera();
            onClose();
        }, "image/jpeg", 0.9);
    };

    const handleClose = () => {
        stopCamera();
        onClose();
    };

    return (
        <Dialog open={isOpen} onOpenChange={handleClose}>
            <DialogContent className="max-w-4xl max-h-[90vh] overflow-auto">
                <DialogHeader>
                    <DialogTitle>Chụp ảnh</DialogTitle>
                </DialogHeader>

                {error ? (
                    <div className="flex flex-col items-center justify-center p-8 text-center">
                        <X className="h-12 w-12 text-destructive mb-4" />
                        <p className="text-destructive mb-4">{error}</p>
                        <Button onClick={handleClose}>Đóng</Button>
                    </div>
                ) : (
                    <>
                        <div className="relative w-full bg-black rounded-lg overflow-hidden">
                            <video
                                ref={videoRef}
                                autoPlay
                                playsInline
                                className="w-full h-auto"
                                style={{ maxHeight: "70vh" }}
                            />
                            <div className="absolute inset-0 pointer-events-none border-4 border-white/50" />
                        </div>

                        <DialogFooter>
                            <Button variant="outline" onClick={handleClose} disabled={isCapturing}>
                                Hủy
                            </Button>
                            <Button
                                onClick={capturePhoto}
                                disabled={isCapturing}
                                className="bg-primary hover:bg-primary/90"
                            >
                                {isCapturing ? (
                                    "Đang xử lý..."
                                ) : (
                                    <>
                                        <Camera className="w-4 h-4 mr-2" />
                                        Chụp ảnh
                                    </>
                                )}
                            </Button>
                        </DialogFooter>
                    </>
                )}
            </DialogContent>
        </Dialog>
    );
}


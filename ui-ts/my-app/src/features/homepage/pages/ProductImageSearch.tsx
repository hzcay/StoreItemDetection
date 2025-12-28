import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { SearchResult, SearchResponse } from "@/features/product/types/product.search.type";
import { searchProductsByImage } from "@/features/product/services/product.image.api";
import { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, Search, Tag, AlertCircle, Sparkles } from "lucide-react";

export function ProductImageSearch() {
    const [uploadedImage, setUploadedImage] = useState<string | null>(null);
    const [searchResponse, setSearchResponse] = useState<SearchResponse | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const navigate = useNavigate();

    useEffect(() => {
        const searchByImage = async () => {
            const imageData = sessionStorage.getItem('uploadedImage');
            if (!imageData) {
                setError('Không tìm thấy ảnh trong bộ nhớ tạm. Vui lòng thử lại.');
                return;
            }

            try {
                setIsLoading(true);
                setError(null);
                setUploadedImage(imageData);

                const response = await fetch(imageData);
                const blob = await response.blob();
                const file = new File([blob], 'uploaded-image.jpg', { type: 'image/jpeg' });

                const searchResponse = await searchProductsByImage(file);
                setSearchResponse(searchResponse);

                // Draw bounding boxes for exact matches only
                setTimeout(() => drawBoundingBoxes(imageData, searchResponse.results), 150);
            } catch (err) {
                console.error('Error searching by image:', err);
                setError('Không thể kết nối đến máy chủ phân tích hình ảnh.');
            } finally {
                setIsLoading(false);
            }
        };

        searchByImage();
    }, []);

    const drawBoundingBoxes = (imageSrc: string, results: SearchResult[]) => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const img = new Image();
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;

            const ctx = canvas.getContext('2d');
            if (!ctx) return;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);

            results.forEach((result, index) => {
                if (!result.bbox) return;
                const { x1, y1, x2, y2 } = result.bbox;

                const labelText = `#${index + 1} ${result.product.name}`;
                const scoreText = `${result.similarity_percent.toFixed(1)}%`;
                const fullText = `${labelText} (${scoreText})`;

                // 1. GIẢM ĐỘ DÀY VIỀN: Sử dụng tỷ lệ nhỏ hơn (0.002 thay vì 0.005)
                ctx.strokeStyle = '#22c55e';
                ctx.lineWidth = Math.max(2, img.width * 0.0025);
                ctx.lineJoin = 'round';
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                // 2. GIẢM SIZE CHỮ: Sử dụng tỷ lệ 0.012 thay vì 0.02
                const fontSize = Math.max(12, img.width * 0.015);
                ctx.font = `500 ${fontSize}px Inter, system-ui, sans-serif`;

                const paddingX = fontSize * 0.6;
                const paddingY = fontSize * 0.3;
                const metrics = ctx.measureText(fullText);
                const bgWidth = metrics.width + paddingX * 2;
                const bgHeight = fontSize + paddingY * 2;

                const labelY = (y1 - bgHeight < 0) ? y1 : y1 - bgHeight;

                // Vẽ nền nhãn mảnh mai hơn
                ctx.fillStyle = '#22c55e';
                ctx.fillRect(x1, labelY, bgWidth, bgHeight);

                // Vẽ chữ nhỏ gọn
                ctx.fillStyle = 'white';
                ctx.textBaseline = 'top';
                ctx.fillText(fullText, x1 + paddingX, labelY + paddingY);
            });
        };
        img.src = imageSrc;
    };

    if (isLoading) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[70vh] space-y-6">
                <div className="w-full max-w-md px-10">
                    <Progress value={80} className="h-1.5 animate-pulse" />
                </div>
                <div className="text-center">
                    <h3 className="text-lg font-medium text-primary">AI đang xử lý...</h3>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="container mx-auto px-4 py-20 flex justify-center">
                <div className="bg-destructive/5 border border-destructive/10 p-6 rounded-xl max-w-md text-center">
                    <AlertCircle className="w-10 h-10 text-destructive mx-auto mb-4" />
                    <p className="text-muted-foreground mb-6 text-sm">{error}</p>
                    <Button variant="outline" onClick={() => navigate(-1)}>Quay lại</Button>
                </div>
            </div>
        );
    }

    return (
        <div className="container mx-auto px-4 py-8 max-w-7xl">
            <header className="mb-8 flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight mb-1">Kết quả tìm kiếm</h1>
                    {searchResponse?.has_exact_match ? (
                        <p className="text-sm text-muted-foreground">
                            Tìm thấy {searchResponse.results.length} sản phẩm khớp.
                        </p>
                    ) : (
                        <p className="text-sm text-muted-foreground">
                            Không tìm thấy sản phẩm khớp. Dưới đây là các gợi ý tương tự.
                        </p>
                    )}
                </div>
                <Button size="sm" variant="ghost" onClick={() => navigate(-1)}>Tải ảnh khác</Button>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">

                {/* ẢNH LỚN */}
                <div className="lg:col-span-8 w-full" ref={containerRef}>
                    <div className="sticky top-24">
                        <div className="relative rounded-2xl overflow-hidden shadow-xl border border-slate-200 bg-slate-50">
                            <canvas
                                ref={canvasRef}
                                className="w-full h-auto block"
                            />
                        </div>
                    </div>
                </div>

                {/* DANH SÁCH BÊN PHẢI */}
                <div className="lg:col-span-4 space-y-6">
                    {/* Exact Matches */}
                    {searchResponse && searchResponse.results.length > 0 && (
                        <div>
                            <div className="flex items-center gap-2 mb-4 text-sm font-semibold text-slate-500 uppercase tracking-wider">
                                <Tag className="w-4 h-4" />
                                Sản phẩm tìm thấy ({searchResponse.results.length})
                            </div>
                            <div className="flex flex-col gap-3 max-h-[40vh] overflow-y-auto pr-2 custom-scrollbar">
                                {searchResponse.results.map((result, index) => (
                                    <Card key={`exact-${result.product.id}`} className="hover:shadow-md transition-shadow duration-200 border-green-200 shadow-sm">
                                        <CardContent className="p-4">
                                            <div className="flex gap-4">
                                                <div className="flex-shrink-0 w-10 h-10 rounded-full bg-green-500 text-white flex items-center justify-center font-bold text-sm shadow-sm">
                                                    {index + 1}
                                                </div>
                                                <div className="flex-grow min-w-0">
                                                    <div className="flex justify-between items-start mb-1 gap-2">
                                                        <h4 className="font-semibold text-base truncate leading-tight">
                                                            {result.product.name}
                                                        </h4>
                                                        <Badge variant="outline" className="text-green-600 border-green-200 bg-green-50 shrink-0 text-[10px] h-5">
                                                            {result.similarity_percent.toFixed(1)}%
                                                        </Badge>
                                                    </div>
                                                    <p className="text-[10px] text-muted-foreground mb-3 font-mono">
                                                        ID: {result.product.id}
                                                    </p>
                                                    <Button
                                                        size="sm"
                                                        className="w-full bg-slate-900 hover:bg-green-600 text-white text-xs h-8"
                                                        onClick={() => window.location.href = `http://localhost:5173/products/${result.product.id}`}
                                                    >
                                                        Xem chi tiết
                                                        <ArrowRight className="ml-2 w-3 h-3" />
                                                    </Button>
                                                </div>
                                            </div>
                                        </CardContent>
                                    </Card>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Suggested Products */}
                    {searchResponse && searchResponse.suggested_products.length > 0 && (
                        <div>
                            <div className="flex items-center gap-2 mb-4 text-sm font-semibold text-slate-500 uppercase tracking-wider">
                                <Sparkles className="w-4 h-4" />
                                Sản phẩm tương tự ({searchResponse.suggested_products.length})
                            </div>
                            <div className="flex flex-col gap-3 max-h-[40vh] overflow-y-auto pr-2 custom-scrollbar">
                                {searchResponse.suggested_products.map((result, index) => (
                                    <Card key={`suggested-${result.product.id}`} className="hover:shadow-md transition-shadow duration-200 border-slate-200 shadow-sm">
                                        <CardContent className="p-4">
                                            <div className="flex gap-4">
                                                <div className="flex-shrink-0 w-10 h-10 rounded-full bg-slate-400 text-white flex items-center justify-center font-bold text-sm shadow-sm">
                                                    {index + 1}
                                                </div>
                                                <div className="flex-grow min-w-0">
                                                    <div className="flex justify-between items-start mb-1 gap-2">
                                                        <h4 className="font-semibold text-base truncate leading-tight">
                                                            {result.product.name}
                                                        </h4>
                                                        <Badge variant="outline" className="text-slate-600 border-slate-200 bg-slate-50 shrink-0 text-[10px] h-5">
                                                            {result.similarity_percent.toFixed(1)}%
                                                        </Badge>
                                                    </div>
                                                    <p className="text-[10px] text-muted-foreground mb-3 font-mono">
                                                        ID: {result.product.id}
                                                    </p>
                                                    <Button
                                                        size="sm"
                                                        className="w-full bg-slate-700 hover:bg-slate-600 text-white text-xs h-8"
                                                        onClick={() => window.location.href = `http://localhost:5173/products/${result.product.id}`}
                                                    >
                                                        Xem chi tiết
                                                        <ArrowRight className="ml-2 w-3 h-3" />
                                                    </Button>
                                                </div>
                                            </div>
                                        </CardContent>
                                    </Card>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* No Results Message */}
                    {searchResponse && !searchResponse.has_exact_match && searchResponse.suggested_products.length === 0 && (
                        <div className="text-center py-8 text-muted-foreground">
                            <AlertCircle className="w-12 h-12 mx-auto mb-4 opacity-50" />
                            <p>Không tìm thấy sản phẩm nào.</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
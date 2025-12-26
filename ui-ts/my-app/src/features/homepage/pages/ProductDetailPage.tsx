import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { Product } from "@/features/product/types/types";
import { productApi } from "@/features/product/services/product.api";
import { Button } from "@/components/ui/button";
import { ShoppingCart, ArrowLeft } from "lucide-react";
import { toast } from "sonner";

export function ProductDetailPage() {
    const { id } = useParams<{ id: string }>();
    const [product, setProduct] = useState<Product | null>(null);
    const [loading, setLoading] = useState(true);
    const [selectedImage, setSelectedImage] = useState(0);

    useEffect(() => {
        if (!id) return;

        const fetchProduct = async () => {
            try {
                setLoading(true);
                const data = await productApi.getById(Number(id));
                setProduct(data);
            } catch (error) {
                console.error(error);
                toast.error("Failed to load product details.");
            } finally {
                setLoading(false);
            }
        };

        fetchProduct();
    }, [id]);

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary" />
            </div>
        );
    }

    if (!product) {
        return (
            <div className="text-center py-12">
                <h2 className="text-2xl font-bold mb-4">Product not found</h2>
                <Button asChild>
                    <Link to="/">Back to Products</Link>
                </Button>
            </div>
        );
    }

    // Prefer primary image
    const images = product.images ?? [];
    const primaryImage =
        images.find(img => img.is_primary) ?? images[selectedImage];

    return (
        <div className="container mx-auto px-4 py-8">
            <Button variant="ghost" asChild className="mb-6">
                <Link to="/" className="flex items-center gap-2">
                    <ArrowLeft className="h-4 w-4" />
                    Back to Products
                </Link>
            </Button>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Images */}
                <div>
                    <div className="bg-white rounded-lg overflow-hidden border mb-4">
                        {primaryImage ? (
                            <img
                                src={primaryImage.image_url}
                                alt={product.name}
                                className="w-full max-h-[500px] object-contain"
                            />
                        ) : (
                            <div className="w-full aspect-square bg-gray-100 flex items-center justify-center">
                                <span className="text-gray-400">
                                    No image available
                                </span>
                            </div>
                        )}
                    </div>

                    {images.length > 1 && (
                        <div className="flex gap-2 overflow-x-auto py-2">
                            {images.map((image, index) => (
                                <button
                                    key={image.id}
                                    onClick={() => setSelectedImage(index)}
                                    className={`w-16 h-16 rounded overflow-hidden border-2 ${selectedImage === index
                                        ? "border-primary"
                                        : "border-transparent"
                                        }`}
                                >
                                    <img
                                        src={image.image_url}
                                        alt={`${product.name} ${index + 1}`}
                                        className="w-full h-full object-cover"
                                    />
                                </button>
                            ))}
                        </div>
                    )}
                </div>

                {/* Info */}
                <div className="space-y-6">
                    <div>
                        <h1 className="text-3xl font-bold mb-2">
                            {product.name}
                        </h1>
                        {product.category && (
                            <span className="text-sm text-muted-foreground">
                                Category: {product.category.name}
                            </span>
                        )}
                    </div>

                    <div className="flex items-center gap-4">
                        <span className="text-3xl font-bold">
                            ${product.price.toLocaleString()}
                        </span>
                        <span
                            className={`px-3 py-1 rounded-full text-sm font-medium ${product.stock_quantity > 0
                                ? "bg-green-100 text-green-800"
                                : "bg-red-100 text-red-800"
                                }`}
                        >
                            {product.stock_quantity > 0
                                ? "In Stock"
                                : "Out of Stock"}
                        </span>
                    </div>

                    {product.description && (
                        <div>
                            <h3 className="text-lg font-medium mb-2">
                                Description
                            </h3>
                            <p className="text-muted-foreground">
                                {product.description}
                            </p>
                        </div>
                    )}

                    <div className="grid grid-cols-2 gap-4 pt-4">
                        <div>
                            <p className="text-sm text-muted-foreground">SKU</p>
                            <p className="font-medium">{product.sku ?? "N/A"}</p>
                        </div>
                        <div>
                            <p className="text-sm text-muted-foreground">
                                Barcode
                            </p>
                            <p className="font-mono">
                                {product.barcode ?? "N/A"}
                            </p>
                        </div>
                        <div>
                            <p className="text-sm text-muted-foreground">
                                Status
                            </p>
                            <p className="font-medium">
                                {product.is_active ? "Active" : "Inactive"}
                            </p>
                        </div>
                        <div>
                            <p className="text-sm text-muted-foreground">
                                Stock
                            </p>
                            <p className="font-medium">
                                {product.stock_quantity} units
                            </p>
                        </div>
                    </div>

                    <Button
                        size="lg"
                        className="w-full"
                        disabled={product.stock_quantity === 0}
                    >
                        <ShoppingCart className="mr-2 h-4 w-4" />
                        Add to Cart
                    </Button>
                </div>
            </div>
        </div>
    );
}

export default ProductDetailPage;

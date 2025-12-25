import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Product } from "@/features/product/types/types";
import { Link } from "react-router-dom";

interface ProductCardProps {
    product: Product;
}

export function ProductCard({ product }: ProductCardProps) {
    const primaryImage = product.images?.[0];
    const getImageUrl = (path: string) => {
        if (!path) return null;
        const cleanPath = path.startsWith('/') ? path.slice(1) : path;
        return `http://localhost:8000/${cleanPath}`;
    };
    console.log(primaryImage?.image_url)
    const imageUrl = getImageUrl(primaryImage?.image_url);
    console.log(imageUrl)

    return (
        <Card className="w-full max-w-sm overflow-hidden transition-transform hover:shadow-lg hover:-translate-y-1">
            <Link to={`/products/${product.id}`}>
                <div className="aspect-square overflow-hidden">
                    {imageUrl ? (
                        <img
                            src={imageUrl}
                            alt={product.name}
                            className="w-full h-full object-cover hover:scale-105 transition-transform duration-300"
                            onError={(e) => {
                                try {
                                    const target = e.target as HTMLImageElement;
                                    target.onerror = null; // Prevent infinite loop
                                    target.style.display = 'none';

                                    const parent = target.parentElement;
                                    if (parent) {
                                        const placeholder = document.createElement('div');
                                        placeholder.className = 'w-full h-full bg-gray-100 flex items-center justify-center';
                                        const span = document.createElement('span');
                                        span.className = 'text-gray-400';
                                        span.textContent = 'Image not available';
                                        placeholder.appendChild(span);
                                        parent.appendChild(placeholder);
                                    }
                                } catch (error) {
                                    console.error('Error handling image load error:', error);
                                }
                            }}
                        />
                    ) : (
                        <div className="w-full h-full bg-gray-100 flex items-center justify-center">
                            <span className="text-gray-400">No image</span>
                        </div>
                    )}
                </div>
            </Link>
            <CardHeader className="pb-2">
                <CardTitle className="text-lg font-medium line-clamp-1">{product.name}</CardTitle>
            </CardHeader>
            <CardContent className="pb-2">
                <div className="flex justify-between items-center">
                    <span className="text-2xl font-bold">${product.price.toLocaleString()}</span>
                    <span className={`text-sm px-2 py-1 rounded-full ${product.stock_quantity > 0 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                        {product.stock_quantity > 0 ? 'In Stock' : 'Out of Stock'}
                    </span>
                </div>
            </CardContent>
            <CardFooter className="flex justify-between">
                <Button variant="outline" asChild>
                    <Link to={`/products/${product.id}`}>
                        View Details
                    </Link>
                </Button>
                <Button disabled={product.stock_quantity === 0}>
                    Add to Cart
                </Button>
            </CardFooter>
        </Card>
    );
}

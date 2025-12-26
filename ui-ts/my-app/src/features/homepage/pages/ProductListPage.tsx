import { useEffect, useState } from "react";
import { productApi } from "@/features/product/services/product.api";
import { Product } from "@/features/product/types/types";
import { ProductCard } from "../components/ProductCard";
import { ProductSearch } from "../components/ProductSearch";
import Layout from "@/components/layouts/Layout";

export function ProductListPage() {
    const [products, setProducts] = useState<Product[]>([]);
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState("");

    useEffect(() => {
        const fetchProducts = async () => {
            try {
                setLoading(true);
                const data = await productApi.list();
                setProducts(data);
            } catch (error) {
                console.error("Error fetching products:", error);
            } finally {
                setLoading(false);
            }
        };

        fetchProducts();
    }, []);

    const filteredProducts = products.filter((product) =>
        product.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        product.description?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        product.category?.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
            </div>
        );
    }

    return (
        <Layout>
            <div className="container mx-auto px-4 py-8">
                <div className="flex flex-col md:flex-row justify-between items-start mb-8 gap-4">
                    <h1 className="text-3xl font-bold">Our Products</h1>

                    {/* 🔍 Integrated Search */}
                    <ProductSearch
                        onTextSearch={(text) => setSearchTerm(text)}
                        onImageUpload={(file) => {
                            console.log("Uploaded image:", file);
                            // Convert the file to base64 and store it in sessionStorage
                            const reader = new FileReader();
                            reader.onloadend = () => {
                                const base64String = reader.result as string;
                                sessionStorage.setItem('uploadedImage', base64String);
                                // Navigate to the image search page
                                window.location.href = '/products-image-search';
                            };
                            reader.readAsDataURL(file);
                        }}
                    />
                </div>

                {filteredProducts.length === 0 ? (
                    <div className="text-center py-12">
                        <p className="text-lg text-muted-foreground">
                            No products found.
                        </p>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
                        {filteredProducts.map((product) => (
                            <ProductCard key={product.id} product={product} />
                        ))}
                    </div>
                )}
            </div>
        </Layout>
    );
}

export default ProductListPage;

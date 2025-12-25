// src/features/product/page/Page.tsx
import { useState } from "react";
import Cropper from "react-easy-crop";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogFooter } from "@/components/ui/dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Pencil, Trash2, Loader2, Plus, Search, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import Layout from "@/components/layouts/Layout";
import { toast } from "sonner";

import { useProducts, useCreateProduct, useUpdateProduct, useDeleteProduct } from "../hooks/useProducts";
import { useCategories } from "@/features/category/hooks/useCategories";
import { Product, CreateProductDto, UpdateProductDto } from "../types/types";
import { getCroppedImg } from "@/util/cropImage";
import ProductForm from "../components/ProductForm";

/* ========================= PAGE ========================= */

export default function ProductPage() {
    const createProduct = useCreateProduct();
    const updateProduct = useUpdateProduct();
    const deleteProduct = useDeleteProduct();

    const { data: products, isLoading, error, refetch } = useProducts();
    const { data: categories } = useCategories();

    const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
    const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
    const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);
    const [searchTerm, setSearchTerm] = useState("");

    const [formData, setFormData] = useState<CreateProductDto>({
        name: "",
        description: "",
        price: 0,
        stock_quantity: 0,
        sku: "",
        barcode: "",
        is_active: true,
        category_id: undefined,
        images: []
    });

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            if (selectedProduct) {
                await updateProduct.mutateAsync({
                    id: selectedProduct.id,
                    data: formData as UpdateProductDto
                });
                toast.success("Product updated");
            } else {
                await createProduct.mutateAsync(formData);
                toast.success("Product created");
            }

            setIsCreateDialogOpen(false);
            setIsEditDialogOpen(false);
            setSelectedProduct(null);
            setFormData({
                name: "",
                description: "",
                price: 0,
                stock_quantity: 0,
                sku: "",
                barcode: "",
                is_active: true,
                category_id: undefined,
                images: []
            });

            refetch();
        } catch {
            toast.error("Something went wrong");
        }
    };

    const filteredProducts =
        products?.filter(p =>
            p.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
            p.sku.toLowerCase().includes(searchTerm.toLowerCase())
        ) || [];

    if (isLoading) {
        return (
            <Layout>
                <div className="flex justify-center p-10">
                    <Loader2 className="animate-spin" />
                </div>
            </Layout>
        );
    }

    if (error) {
        return <Layout>Error loading products</Layout>;
    }

    return (
        <Layout>
            <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white p-6 space-y-6">
                <div className="flex justify-between">
                    <h1 className="text-2xl font-semibold">Products</h1>
                    <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen} modal={false}>
                        <DialogTrigger asChild>
                            <Button>
                                <Plus className="mr-2 h-4 w-4" /> Add Product
                            </Button>
                        </DialogTrigger>
                        <DialogContent className="max-w-2xl bg-white">
                            <DialogHeader>
                                <DialogTitle>Create Product</DialogTitle>
                            </DialogHeader>
                            <ProductForm
                                formData={formData}
                                setFormData={setFormData}
                                categories={categories || []}
                                onSubmit={handleSubmit}
                                isSubmitting={createProduct.isPending}
                            />
                        </DialogContent>
                    </Dialog>
                </div>

                <Card className="bg-white shadow-lg">
                    <CardHeader className="flex justify-between">
                        <CardTitle>Product List</CardTitle>
                        <Input
                            placeholder="Search..."
                            className="w-64"
                            value={searchTerm}
                            onChange={e => setSearchTerm(e.target.value)}
                        />
                    </CardHeader>
                    <CardContent>
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>Name</TableHead>
                                    <TableHead>SKU</TableHead>
                                    <TableHead>Status</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {filteredProducts.map(p => (
                                    <TableRow key={p.id}>
                                        <TableCell>{p.name}</TableCell>
                                        <TableCell>{p.sku}</TableCell>
                                        <TableCell>
                                            <Badge>{p.is_active ? "Active" : "Inactive"}</Badge>
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </CardContent>
                </Card>
            </div>
        </Layout>
    );
}

/* ========================= FORM ========================= */

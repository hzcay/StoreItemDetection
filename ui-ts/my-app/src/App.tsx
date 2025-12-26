import { BrowserRouter, Routes, Route, Navigate, Outlet } from "react-router-dom"
import { ProductListPage } from "@/features/homepage/pages/ProductListPage"
import { ProductDetailPage } from "@/features/homepage/pages/ProductDetailPage"
import CategoryPage from "@/features/category/page/Page"
import ProductPage from "@/features/product/page/page"
import { Toaster } from "sonner"
import { ProductImageSearch } from "./features/homepage/pages/ProductImageSearch"

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public routes */}
        <Route path="/home" element={<ProductListPage />} />
        <Route path="/products/:id" element={<ProductDetailPage />} />
        <Route path="/categories" element={<CategoryPage />} />
        <Route path="/products" element={<ProductPage />} />
        <Route path="/products-image-search" element={<ProductImageSearch />} />
        <Route path="/" element={<Navigate to="/home" replace />} />
      </Routes>
      <Toaster position="top-right" />
    </BrowserRouter>
  )
}
import { useState } from 'react';
import ProductForm from './ProductForm';
import ProductList from './ProductList';

export default function ProductPage() {
  const [refresh, setRefresh] = useState(0);
  return (
    <div className="grid two">
      <ProductForm onCreated={() => setRefresh((x) => x + 1)} />
      <ProductList refreshSignal={refresh} />
    </div>
  );
}

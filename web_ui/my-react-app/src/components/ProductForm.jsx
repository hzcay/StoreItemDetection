import { useState } from 'react';
import { createProduct } from '../api';

export default function ProductForm({ onCreated }) {
  const [formState, setFormState] = useState({
    name: '',
    description: '',
    price: '',
    stock_quantity: 0,
    sku: '',
    barcode: '',
    is_active: true,
    category_id: '',
  });
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormState((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const handleImages = (e) => {
    setImages(Array.from(e.target.files || []));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const formData = new FormData();
      Object.entries(formState).forEach(([k, v]) => {
        if (v !== '' && v !== null && v !== undefined) {
          formData.append(k, v);
        }
      });
      images.forEach((file) => formData.append('images', file));

      await createProduct(formData);
      setFormState({
        name: '',
        description: '',
        price: '',
        stock_quantity: 0,
        sku: '',
        barcode: '',
        is_active: true,
        category_id: '',
      });
      setImages([]);
      if (onCreated) onCreated();
    } catch (err) {
      setError(err.message || 'Create failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form className="card" onSubmit={handleSubmit}>
      <h2>Create Product</h2>
      {error && <div className="error">{error}</div>}
      <div className="grid two">
        <label>
          Name*
          <input name="name" value={formState.name} onChange={handleChange} required />
        </label>
        <label>
          Price*
          <input name="price" type="number" step="0.01" value={formState.price} onChange={handleChange} required />
        </label>
        <label>
          Stock
          <input name="stock_quantity" type="number" value={formState.stock_quantity} onChange={handleChange} />
        </label>
        <label>
          SKU
          <input name="sku" value={formState.sku} onChange={handleChange} />
        </label>
        <label>
          Barcode
          <input name="barcode" value={formState.barcode} onChange={handleChange} />
        </label>
        <label>
          Category ID
          <input name="category_id" value={formState.category_id} onChange={handleChange} />
        </label>
        <label className="full">
          Description
          <textarea name="description" value={formState.description} onChange={handleChange} rows={3} />
        </label>
        <label>
          Active
          <input name="is_active" type="checkbox" checked={formState.is_active} onChange={handleChange} />
        </label>
        <label className="full">
          Images
          <input type="file" multiple accept="image/*" onChange={handleImages} />
        </label>
      </div>
      <button type="submit" disabled={loading}>
        {loading ? 'Saving...' : 'Create'}
      </button>
    </form>
  );
}

import { useEffect, useState } from 'react';
import { listProducts } from '../api';

export default function ProductList({ refreshSignal }) {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      setLoading(true);
      setError('');
      try {
        const data = await listProducts();
        if (!cancelled) setItems(data || []);
      } catch (err) {
        if (!cancelled) setError(err.message || 'Load failed');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    run();
    return () => {
      cancelled = true;
    };
  }, [refreshSignal]);

  return (
    <div className="card">
      <div className="header-row">
        <h2>Products</h2>
        {loading && <span>Loading...</span>}
      </div>
      {error && <div className="error">{error}</div>}
      <div className="list">
        {items.map((p) => (
          <div key={p.id} className="list-item">
            <div className="title">{p.name}</div>
            <div className="meta">
              <span>Price: {p.price}</span>
              <span>Stock: {p.stock_quantity}</span>
              <span>SKU: {p.sku}</span>
            </div>
            <div className="desc">{p.description}</div>
          </div>
        ))}
        {!loading && items.length === 0 && <div>No products</div>}
      </div>
    </div>
  );
}

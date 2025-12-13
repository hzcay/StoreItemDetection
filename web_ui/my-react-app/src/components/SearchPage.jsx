import { useState } from 'react';
import { searchProductsRerank, searchProducts } from '../api';

export default function SearchPage() {
  const [file, setFile] = useState(null);
  const [topK, setTopK] = useState(20);
  const [dinoLimit, setDinoLimit] = useState(10);
  const [clipLimit, setClipLimit] = useState(3);
  const [dinoWeight, setDinoWeight] = useState(0.55);
  const [clipWeight, setClipWeight] = useState(0.45);
  const [fusionThreshold, setFusionThreshold] = useState(0);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Chọn ảnh để search');
      return;
    }
    setError('');
    setLoading(true);
    try {
      const data = await searchProductsRerank({
        file,
        topK: Number(topK) || 20,
        dinoLimit: Number(dinoLimit) || 10,
        clipLimit: Number(clipLimit) || 3,
        dinoWeight: Number(dinoWeight),
        clipWeight: Number(clipWeight),
        fusionThreshold: Number(fusionThreshold) || 0,
      });
      setResults(data?.results || data || []);
    } catch (err) {
      setError(err.message || 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const handlePlainSearch = async () => {
    if (!file) {
      setError('Chọn ảnh để search');
      return;
    }
    setError('');
    setLoading(true);
    try {
      const data = await searchProducts({ file, topK: Number(topK) || 20 });
      setResults(data?.results || data || []);
    } catch (err) {
      setError(err.message || 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>Image Search (DINOv2 + CLIP)</h2>
      {error && <div className="error">{error}</div>}
      <form className="grid two" onSubmit={handleSubmit}>
        <label className="full">
          Query Image
          <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] || null)} />
        </label>
        <label>
          top_k
          <input type="number" min="1" max="100" value={topK} onChange={(e) => setTopK(e.target.value)} />
        </label>
        <label>
          dino_limit
          <input type="number" min="1" max="100" value={dinoLimit} onChange={(e) => setDinoLimit(e.target.value)} />
        </label>
        <label>
          clip_limit
          <input type="number" min="1" max="100" value={clipLimit} onChange={(e) => setClipLimit(e.target.value)} />
        </label>
        <label>
          dino_weight
          <input
            type="number"
            step="0.05"
            min="0"
            max="1"
            value={dinoWeight}
            onChange={(e) => setDinoWeight(e.target.value)}
          />
        </label>
        <label>
          clip_weight
          <input
            type="number"
            step="0.05"
            min="0"
            max="1"
            value={clipWeight}
            onChange={(e) => setClipWeight(e.target.value)}
          />
        </label>
        <label>
          fusion_threshold
          <input
            type="number"
            step="0.05"
            min="0"
            max="1"
            value={fusionThreshold}
            onChange={(e) => setFusionThreshold(e.target.value)}
          />
        </label>
        <div className="actions">
          <button type="submit" disabled={loading}>
            {loading ? 'Searching...' : 'Search with rerank'}
          </button>
          <button type="button" onClick={handlePlainSearch} disabled={loading}>
            Plain search
          </button>
        </div>
      </form>

      <div className="results">
        {results.map((r, idx) => (
          <div key={idx} className="result-card">
            <div className="title">{r.product_name || r.payload?.product_name || `Product ${r.product_id}`}</div>
            <div className="meta">
              <span>product_id: {r.product_id || r.payload?.product_id}</span>
              {' · '}
              <span>dino: {r.dino_score?.toFixed?.(3)}</span>
              {r.clip_score !== undefined && r.clip_score !== null && (
                <>
                  {' · '}
                  <span>clip: {r.clip_score?.toFixed?.(3)}</span>
                </>
              )}
              {' · '}
              <strong>combined: {r.combined_score?.toFixed?.(3)}</strong>
            </div>
            {r.image_path && (
              <div className="thumb">
                <img src={r.image_path} alt={r.product_name || r.product_id} />
              </div>
            )}
            {r.product_description && <div className="desc">{r.product_description}</div>}
          </div>
        ))}
        {!loading && results.length === 0 && <div>No results</div>}
      </div>
    </div>
  );
}

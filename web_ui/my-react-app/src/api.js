const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function request(path, { method = 'GET', body, headers = {} } = {}) {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method,
    headers,
    body,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Request failed (${res.status}): ${text}`);
  }

  const contentType = res.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    return res.json();
  }
  return res.text();
}

export async function listProducts() {
  return request('/api/products');
}

export async function createProduct(formData) {
  return request('/api/products', {
    method: 'POST',
    body: formData,
  });
}

export async function searchProducts({ file, topK = 20 }) {
  const form = new FormData();
  form.append('file', file);
  const params = new URLSearchParams({ top_k: String(topK) }).toString();
  return request(`/api/products/search?${params}`, {
    method: 'POST',
    body: form,
  });
}

export async function searchProductsRerank({
  file,
  topK = 20,
  dinoLimit = 10,
  clipLimit = 3,
  dinoWeight = 0.55,
  clipWeight = 0.45,
  fusionThreshold = 0,
}) {
  const form = new FormData();
  form.append('file', file);
  const params = new URLSearchParams({
    top_k: String(topK),
    dino_limit: String(dinoLimit),
    clip_limit: String(clipLimit),
    dino_weight: String(dinoWeight),
    clip_weight: String(clipWeight),
    fusion_threshold: String(fusionThreshold),
  }).toString();
  return request(`/api/products/search-rerank?${params}`, {
    method: 'POST',
    body: form,
  });
}


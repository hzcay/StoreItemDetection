import { useState } from 'react';
import './App.css';
import ProductPage from './components/ProductPage';
import SearchPage from './components/SearchPage';

function App() {
  const [tab, setTab] = useState('products');

  return (
    <div className="app">
      <header className="topbar">
        <h1>Store Item UI</h1>
        <nav>
          <button className={tab === 'products' ? 'active' : ''} onClick={() => setTab('products')}>
            Products
          </button>
          <button className={tab === 'search' ? 'active' : ''} onClick={() => setTab('search')}>
            Search
          </button>
        </nav>
      </header>
      <main>
        {tab === 'products' ? <ProductPage /> : <SearchPage />}
      </main>
    </div>
  );
}

export default App;

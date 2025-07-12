import { useState } from 'react';
import axios from 'axios';

function Test() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setResult(null);
    setLoading(true);
    const formData = new FormData();
    formData.append('dataset', file);
    try {
      const res = await axios.post(
        'http://localhost:5555/v1/ml/test',
        formData,
        { withCredentials: true }
      );
      setResult(res.data);
    } catch (err) {
      setError('❌ Test failed. ' + (err.response?.data?.error || ''));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page" style={{ minHeight: '80vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', justifyItems: 'center' }}>
      <h2>Test Global Model</h2>
      {!loading && !result && (
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
          <input
            type="file"
            accept=".csv"
            onChange={e => setFile(e.target.files[0])}
            required
          />
          <button type="submit">Submit for Testing</button>
        </form>
      )}
      {loading && (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <div className="circular-loader" style={{ width: 60, height: 60, border: '6px solid #eee', borderTop: '6px solid #007bff', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
          <p style={{ marginTop: 16 }}>Testing in progress...</p>
        </div>
      )}
      {result && (
        <div style={{ textAlign: 'center', maxWidth: 400, background: '#f9f9f9', borderRadius: 12, padding: 24, boxShadow: '0 2px 8px #0001' }}>
          <h3 style={{ color: '#111' }}>Test Results</h3>
          <div style={{ fontSize: 18, margin: '12px 0', color: '#111' }}>
            <div><b>Total Samples:</b> {result.total}</div>
            <div><b>Frauds Detected:</b> {result.frauds}</div>
            <div><b>Fraud Percentage:</b> {result.percent_fraud.toFixed(2)}%</div>
            {result.accuracy !== undefined && (
              <div><b>Accuracy:</b> {(result.accuracy * 100).toFixed(2)}%</div>
            )}
          </div>
          <button style={{ marginTop: 20 }} onClick={() => { setResult(null); setFile(null); }}>Test Another File</button>
        </div>
      )}
      {error && <p style={{ color: 'red', marginTop: 16 }}>{error}</p>}
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

export default Test; 
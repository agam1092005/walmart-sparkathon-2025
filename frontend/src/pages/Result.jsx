import { useEffect, useState } from 'react';
import axios from 'axios';

function ResultPage() {
  const [result, setResult] = useState(null);

  useEffect(() => {
    axios.get('http://localhost:5000/result')
      .then((res) => setResult(res.data))
      .catch((err) => console.error(err));
  }, []);

  return (
    <div className="page">
      <h2>Fraud Detection Result</h2>
      {result ? (
        <div className="result-display">
          <p><strong>Score:</strong> {result.score}</p>
          <p><strong>Status:</strong> {result.label}</p>
        </div>
      ) : (
        <p>Loading result...</p>
      )}
    </div>
  );
}

export default ResultPage;
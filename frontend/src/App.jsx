import React, { useState } from 'react';
import axios from 'axios';

export default function App() {
  // Initialize state for 20 features
  const [features, setFeatures] = useState(Array(20).fill(0.0));
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (index, value) => {
    const newFeatures = [...features];
    newFeatures[index] = parseFloat(value) || 0.0;
    setFeatures(newFeatures);
  };

  const loadRandomSample = () => {
    // Generates random floats between -2 and 2 for a quick test
    const randomVals = Array.from({ length: 20 }, () => parseFloat((Math.random() * 4 - 2).toFixed(2)));
    setFeatures(randomVals);
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post('http://localhost:8000/predict', {
        features: features
      });
      setResult(response.data);
    } catch (err) {
      setError('Failed to connect to the backend API. Make sure FastAPI is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '800px', margin: '40px auto', fontFamily: 'sans-serif', padding: '20px' }}>
      <h1>ML Model Prediction Dashboard</h1>
      <p>Enter 20 feature values below to run a prediction through your FastAPI backend.</p>

      <button 
        onClick={loadRandomSample} 
        style={{ marginBottom: '20px', padding: '10px 15px', cursor: 'pointer', background: '#4f46e5', color: '#fff', border: 'none', borderRadius: '4px' }}
      >
        Load Random Sample
      </button>

      <form onSubmit={handlePredict}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '15px', marginBottom: '20px' }}>
          {features.map((val, index) => (
            <div key={index}>
              <label style={{ display: 'block', fontSize: '12px', marginBottom: '5px', fontWeight: 'bold' }}>
                Feature {index + 1}
              </label>
              <input
                type="number"
                step="any"
                value={val}
                onChange={(e) => handleInputChange(index, e.target.value)}
                style={{ width: '100%', padding: '8px', boxSizing: 'border-box' }}
              />
            </div>
          ))}
        </div>

        <button 
          type="submit" 
          disabled={loading}
          style={{ width: '100%', padding: '12px', background: '#16a34a', color: '#fff', border: 'none', borderRadius: '4px', fontSize: '16px', cursor: 'pointer' }}
        >
          {loading ? 'Running Prediction...' : 'Run Prediction'}
        </button>
      </form>

      {error && (
        <div style={{ marginTop: '20px', padding: '15px', background: '#fee2e2', color: '#991b1b', borderRadius: '4px' }}>
          {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: '20px', padding: '20px', background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: '4px' }}>
          <h3>Prediction Results</h3>
          <p><strong>Predicted Class:</strong> {result.prediction}</p>
          <p><strong>Confidence Probability:</strong> {(result.probability * 100).toFixed(2)}%</p>
        </div>
      )}
    </div>
  );
}

import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, analyzing, success
  const [result, setResult] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setStatus('analyzing');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE_URL}/analyze`, formData);
      setResult(response.data);
      setStatus('success');
    } catch (error) {
      console.error('Analysis error:', error);
      setStatus('idle');
      alert("Error connecting to backend model.");
    }
  };

  const reset = () => {
    setFile(null);
    setResult(null);
    setStatus('idle');
  };

  return (
    <div className="app-container">
      <div className="logo-header">DeepVoiceAI</div>

      <main className="main-content">
        {status !== 'success' ? (
          <>
            <div 
              className="upload-box"
              onClick={() => fileInputRef.current.click()}
            >
              {file ? file.name : "Upload audio file (.wav / .mp3)"}
            </div>

            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileChange} 
              style={{ display: 'none' }} 
              accept=".wav,.mp3"
            />

            <button 
              className="analyze-button"
              disabled={!file || status === 'analyzing'}
              onClick={handleAnalyze}
            >
              {status === 'analyzing' ? "Analyzing..." : "Analyze Audio"}
            </button>
          </>
        ) : (
          <div className="result-container-wrap">
            <div className="result-stack">
              <div className="result-bar">
                <div className="result-label">FAKE</div>
                <div className="confidence-text">
                  confidence : {result.label === 'FAKE' ? result.confidence : (100 - result.confidence).toFixed(2)}%
                </div>
              </div>

              <div className="result-bar">
                <div className="result-label">REAL</div>
                <div className="confidence-text">
                  confidence : {result.label === 'REAL' ? result.confidence : (100 - result.confidence).toFixed(2)}%
                </div>
              </div>
            </div>

            <button className="reset-button" onClick={reset}>
              back to upload
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;

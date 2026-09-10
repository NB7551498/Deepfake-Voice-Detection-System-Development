import React, { useState } from 'react';
import axios from 'axios';
import { UploadCloud, AlertTriangle, CheckCircle, Activity, Link as LinkIcon } from 'lucide-react';
import './App.css';

function App() {
  const [inputMode, setInputMode] = useState('file'); // 'file' or 'link'
  const [file, setFile] = useState(null);
  const [url, setUrl] = useState('');
  const [detectMode, setDetectMode] = useState('fast'); // 'fast' or 'deep'
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async (e) => {
    if (e) e.preventDefault();
    if (inputMode === 'file' && !file) return;
    if (inputMode === 'link' && (!url || url.trim().length < 5)) return;

    setLoading(true);
    setResult(null);

    const API_BASE = "http://127.0.0.1:8001";

    try {
      let res;
      if (inputMode === 'link') {
        res = await axios.post(`${API_BASE}/predict-url`, {
          url: url.trim(),
          mode: detectMode
        });
      } else {
        const formData = new FormData();
        formData.append("file", file);
        res = await axios.post(`${API_BASE}/predict?mode=${detectMode}`, formData);
      }
      setResult(res.data);
    } catch (err) {
      console.error(err);
      const msg = err.response?.data?.detail || err.message || "Error analyzing audio.";
      alert(`Error: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6 font-sans flex flex-col items-center">
      <div className="max-w-2xl w-full border border-gray-700 rounded-xl overflow-hidden bg-gray-800 shadow-2xl">
        <div className="bg-gray-950 p-4 border-b border-gray-700 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Activity className="text-blue-400" />
            <h1 className="text-lg font-bold tracking-wider text-gray-200">DEEPFAKE VOICE DETECTOR 2.0</h1>
          </div>
          <span className="text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
            Ensemble-v2.0
          </span>
        </div>

        <div className="p-6 space-y-5">
          {/* Mode Selector */}
          <div className="grid grid-cols-2 gap-3">
            <button
              type="button"
              onClick={() => setDetectMode('fast')}
              className={`p-3 rounded-lg border text-left transition-all ${
                detectMode === 'fast'
                  ? 'border-blue-500 bg-blue-500/10'
                  : 'border-gray-700 hover:border-gray-600 bg-gray-900/40'
              }`}
            >
              <div className="font-bold text-xs text-blue-400">⚡ FAST SCAN (&lt;30ms)</div>
              <div className="text-[11px] text-gray-400 mt-0.5">228 Acoustic Features</div>
            </button>
            <button
              type="button"
              onClick={() => setDetectMode('deep')}
              className={`p-3 rounded-lg border text-left transition-all ${
                detectMode === 'deep'
                  ? 'border-blue-500 bg-blue-500/10'
                  : 'border-gray-700 hover:border-gray-600 bg-gray-900/40'
              }`}
            >
              <div className="font-bold text-xs text-blue-400">🔬 DEEP VERIFY (Fusion 2.0)</div>
              <div className="text-[11px] text-gray-400 mt-0.5">Acoustic + Frozen SSL (996 dims)</div>
            </button>
          </div>

          {/* Input Source Tabs */}
          <div className="flex items-center space-x-2 border-b border-gray-700 pb-2">
            <button
              type="button"
              onClick={() => setInputMode('file')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1.5 transition-all ${
                inputMode === 'file'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              <UploadCloud className="w-3.5 h-3.5" />
              <span>Audio File</span>
            </button>
            <button
              type="button"
              onClick={() => setInputMode('link')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1.5 transition-all ${
                inputMode === 'link'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              <LinkIcon className="w-3.5 h-3.5" />
              <span>Audio Link / URL</span>
            </button>
          </div>

          {/* Input Area */}
          {inputMode === 'file' ? (
            <div
              className="border-2 border-dashed border-gray-600 rounded-lg p-6 flex flex-col items-center justify-center hover:border-blue-400 transition-colors cursor-pointer bg-gray-900/30"
              onClick={() => document.getElementById('file-upload').click()}
            >
              <UploadCloud className="w-10 h-10 text-gray-400 mb-2" />
              <span className="text-gray-300 font-medium text-sm">Upload Audio File</span>
              <span className="text-xs text-gray-500 mt-1">{file ? file.name : "WAV, MP3, M4A, FLAC up to 25MB"}</span>
              <input
                id="file-upload"
                type="file"
                className="hidden"
                accept=".wav,.mp3,.m4a,.flac,.ogg"
                onChange={(e) => setFile(e.target.files[0])}
              />
            </div>
          ) : (
            <div className="border border-gray-700 rounded-lg p-4 bg-gray-900/40 space-y-2">
              <label className="block text-xs font-semibold text-gray-300">Enter Audio or Media URL</label>
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com/audio.mp3, YouTube, or SoundCloud link..."
                className="w-full px-3 py-2 rounded-lg bg-gray-950 border border-gray-700 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 font-mono"
              />
              <div className="flex items-center justify-between text-[11px] text-gray-400">
                <span>Direct Audio (MP3, WAV, M4A) or YouTube / SoundCloud</span>
                <span className="text-emerald-400 font-medium">🔒 SSRF Guard</span>
              </div>
            </div>
          )}

          <button
            onClick={handleAnalyze}
            disabled={(inputMode === 'file' && !file) || (inputMode === 'link' && (!url || url.trim().length < 5)) || loading}
            className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-2.5 px-4 rounded-lg transition-colors disabled:opacity-40 text-sm shadow-md"
          >
            {loading ? "Analyzing Audio..." : "Run Detection"}
          </button>
        </div>

        {result && (
          <div className="border-t border-gray-700 p-8 bg-gray-850">
            <h2 className="text-sm text-gray-400 uppercase tracking-wider mb-4">Result</h2>
            
            <div className={`p-4 rounded-lg flex items-center space-x-4 mb-6 ${result.prediction === 'DEEPFAKE' ? 'bg-red-900/30 border border-red-800' : result.prediction === 'REAL' ? 'bg-green-900/30 border border-green-800' : 'bg-yellow-900/30 border border-yellow-800'}`}>
              {result.prediction === 'DEEPFAKE' ? <AlertTriangle className="text-red-500 w-8 h-8" /> : <CheckCircle className="text-green-500 w-8 h-8" />}
              <div>
                <div className={`text-xl font-bold ${result.prediction === 'DEEPFAKE' ? 'text-red-400' : 'text-green-400'}`}>
                  LIKELY {result.prediction}
                </div>
                <div className="text-gray-400 text-sm">Fake probability: {(result.fake_probability * 100).toFixed(1)}%</div>
              </div>
            </div>

            <div className="mt-8">
              <h3 className="text-sm text-gray-400 uppercase tracking-wider mb-4 border-b border-gray-700 pb-2">Segment Analysis</h3>
              <div className="space-y-3">
                {result.segments.map((seg, idx) => (
                  <div key={idx} className="flex justify-between items-center text-sm">
                    <span className="text-gray-400 w-16">{seg.start.toFixed(1)}–{seg.end.toFixed(1)}s</span>
                    <span className={`font-bold w-16 ${seg.fake_probability > 0.5 ? 'text-red-400' : 'text-green-400'}`}>
                      {seg.fake_probability > 0.5 ? 'FAKE' : 'REAL'}
                    </span>
                    <div className="flex-1 mx-4 bg-gray-700 h-2 rounded-full overflow-hidden">
                      <div 
                        className={`h-full ${seg.fake_probability > 0.5 ? 'bg-red-500' : 'bg-green-500'}`} 
                        style={{width: `${seg.fake_probability * 100}%`}}
                      />
                    </div>
                    <span className="text-gray-400 w-12 text-right">{(seg.fake_probability * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

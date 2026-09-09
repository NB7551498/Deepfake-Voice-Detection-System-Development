import React, { useState } from 'react';
import axios from 'axios';
import { UploadCloud, AlertTriangle, CheckCircle, Activity } from 'lucide-react';
import './App.css'; // Assume Tailwind is loaded here

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      // Point to FastAPI backend
      const res = await axios.post("http://localhost:8000/predict", formData);
      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert("Error analyzing audio.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8 font-sans flex flex-col items-center">
      <div className="max-w-2xl w-full border border-gray-700 rounded-xl overflow-hidden bg-gray-800 shadow-2xl">
        <div className="bg-gray-950 p-4 border-b border-gray-700 flex items-center justify-center space-x-3">
          <Activity className="text-blue-400" />
          <h1 className="text-xl font-bold tracking-widest text-gray-200">AI VOICE DETECTOR</h1>
        </div>

        <div className="p-8">
          <div className="border-2 border-dashed border-gray-600 rounded-lg p-8 flex flex-col items-center justify-center hover:border-blue-400 transition-colors cursor-pointer relative"
               onClick={() => document.getElementById('file-upload').click()}>
            <UploadCloud className="w-12 h-12 text-gray-400 mb-4" />
            <span className="text-gray-300 font-medium">Upload Audio</span>
            <span className="text-sm text-gray-500 mt-2">{file ? file.name : "WAV, MP3, M4A up to 10MB"}</span>
            <input 
              id="file-upload" 
              type="file" 
              className="hidden" 
              accept=".wav,.mp3,.m4a" 
              onChange={(e) => setFile(e.target.files[0])} 
            />
          </div>

          <button 
            onClick={handleUpload}
            disabled={!file || loading}
            className="w-full mt-6 bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 px-4 rounded transition-colors disabled:opacity-50"
          >
            {loading ? "Analyzing Voice..." : "Analyze Voice"}
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

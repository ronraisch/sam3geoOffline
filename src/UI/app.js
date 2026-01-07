import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Upload, 
  MousePointer2, 
  Hand, 
  ZoomIn, 
  ZoomOut, 
  Layers, 
  Download, 
  Trash2, 
  RefreshCw,
  BoxSelect,
  AlertCircle
} from 'lucide-react';

// --- API CONFIGURATION ---
const API_BASE = "https://stanford-atom-julie-carlos.trycloudflare.com";

export default function App() {
  // --- STATE ---
  // Application Phase: 'input' | 'labeling'
  const [appState, setAppState] = useState('input');
  
  // Image Data
  const [imageUrl, setImageUrl] = useState('');
  const [imagePath, setImagePath] = useState(null); // Internal backend path
  const [displayImageSrc, setDisplayImageSrc] = useState(null);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });

  // Viewport State (Pan/Zoom)
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Interaction Mode: 'pan' | 'box'
  const [tool, setTool] = useState('box');

  // Box Drawing State
  const [boxes, setBoxes] = useState([]); // [[x1,y1,x2,y2], ...] (Image Coordinates)
  const [currentBox, setCurrentBox] = useState(null); // {x1,y1,x2,y2} (Image Coordinates)

  // Mask/Result State
  const [isSegmenting, setIsSegmenting] = useState(false);
  const [maskImageUrl, setMaskImageUrl] = useState(null);
  const [rawMaskImage, setRawMaskImage] = useState(null); // The actual Image object
  const [threshold, setThreshold] = useState(0.25);
  
  // Loading/Error
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Refs
  const containerRef = useRef(null);
  const maskCanvasRef = useRef(null);

  // --- API FUNCTIONS ---

  const handleUpload = async () => {
    if (!imageUrl) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/upload-path`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: imageUrl }),
      });
      
      if (!res.ok) throw new Error("Failed to upload image path");
      
      const data = await res.json();
      setImagePath(data.internal_path);
      
      // Construct download URL
      const downloadUrl = `${API_BASE}/download/${data.internal_path}`;
      setDisplayImageSrc(downloadUrl);
      
      // Reset State
      setBoxes([]);
      setMaskImageUrl(null);
      setRawMaskImage(null);
      setAppState('labeling');
      setScale(1);
      setOffset({ x: 0, y: 0 });

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSegment = async () => {
    if (boxes.length === 0) {
      setError("Draw at least one box first.");
      return;
    }
    setLoading(true);
    setIsSegmenting(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/predict-boxes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ boxes }),
      });

      if (!res.ok) throw new Error("Segmentation failed");

      const data = await res.json();
      
      // Load the mask image specifically for canvas manipulation
      const maskSrc = `${API_BASE}/download/${data.sum_image_url}`;
      setMaskImageUrl(maskSrc);

      const img = new Image();
      img.crossOrigin = "Anonymous";
      img.src = maskSrc;
      img.onload = () => {
        setRawMaskImage(img);
        setLoading(false);
        setIsSegmenting(false);
      };

    } catch (err) {
      setError(err.message);
      setLoading(false);
      setIsSegmenting(false);
    }
  };

  const handleDownloadGeoData = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/export-geodata?threshold=${threshold}`);
      if (!res.ok) throw new Error("Export failed");
      
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `export_${threshold}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    if(confirm("Are you sure? This will delete all data on the server.")) {
      try {
        await fetch(`${API_BASE}/cleanup`, { method: 'DELETE' });
        window.location.reload();
      } catch (err) {
        setError("Cleanup failed");
      }
    }
  };

  // --- CANVAS / INTERACTION LOGIC ---

  const handleImageLoad = (e) => {
    const { naturalWidth, naturalHeight } = e.target;
    setImageDimensions({ width: naturalWidth, height: naturalHeight });
    
    // Fit image to screen initially
    if (containerRef.current) {
      const { clientWidth, clientHeight } = containerRef.current;
      const scaleX = clientWidth / naturalWidth;
      const scaleY = clientHeight / naturalHeight;
      const initialScale = Math.min(scaleX, scaleY, 1) * 0.9;
      setScale(initialScale);
      
      // Center it
      const newX = (clientWidth - naturalWidth * initialScale) / 2;
      const newY = (clientHeight - naturalHeight * initialScale) / 2;
      setOffset({ x: newX, y: newY });
    }
  };

  // Helper to get Image Coordinates from Mouse Event
  const getCoords = (e) => {
    if (!containerRef.current) return { x: 0, y: 0 };
    const rect = containerRef.current.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const clientY = e.clientY - rect.top;
    
    // Convert Screen -> View -> Image
    const x = (clientX - offset.x) / scale;
    const y = (clientY - offset.y) / scale;
    return { x, y };
  };

  const onMouseDown = (e) => {
    if (tool === 'pan' || e.button === 1) { // Middle click or Pan tool
      setIsDragging(true);
      setDragStart({ x: e.clientX, y: e.clientY });
      return;
    }

    if (tool === 'box') {
      const { x, y } = getCoords(e);
      setCurrentBox({ x1: x, y1: y, x2: x, y2: y });
    }
  };

  const onMouseMove = (e) => {
    if (isDragging) {
      const dx = e.clientX - dragStart.x;
      const dy = e.clientY - dragStart.y;
      setOffset(prev => ({ x: prev.x + dx, y: prev.y + dy }));
      setDragStart({ x: e.clientX, y: e.clientY });
      return;
    }

    if (currentBox) {
      const { x, y } = getCoords(e);
      setCurrentBox(prev => ({ ...prev, x2: x, y2: y }));
    }
  };

  const onMouseUp = () => {
    if (isDragging) {
      setIsDragging(false);
      return;
    }

    if (currentBox) {
      // Normalize box (ensure x1 < x2)
      const x1 = Math.min(currentBox.x1, currentBox.x2);
      const x2 = Math.max(currentBox.x1, currentBox.x2);
      const y1 = Math.min(currentBox.y1, currentBox.y2);
      const y2 = Math.max(currentBox.y1, currentBox.y2);

      // Only add if it has some size
      if (x2 - x1 > 2 && y2 - y1 > 2) {
        setBoxes(prev => [...prev, [x1, y1, x2, y2]]);
      }
      setCurrentBox(null);
    }
  };

  const onWheel = (e) => {
    e.preventDefault();
    const zoomSensitivity = 0.001;
    const delta = -e.deltaY * zoomSensitivity;
    const newScale = Math.min(Math.max(0.1, scale + delta), 20);
    
    // Zoom towards mouse pointer logic
    // 1. Get mouse pos relative to container
    const rect = containerRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // 2. Calculate offset adjustment
    const scaleRatio = newScale / scale;
    const newX = mouseX - (mouseX - offset.x) * scaleRatio;
    const newY = mouseY - (mouseY - offset.y) * scaleRatio;

    setScale(newScale);
    setOffset({ x: newX, y: newY });
  };

  // --- MASK VISUALIZATION EFFECT ---
  // Renders the grayscale result into a canvas, applying color and threshold
  useEffect(() => {
    if (!rawMaskImage || !maskCanvasRef.current) return;

    const canvas = maskCanvasRef.current;
    const ctx = canvas.getContext('2d');
    const w = rawMaskImage.width;
    const h = rawMaskImage.height;

    canvas.width = w;
    canvas.height = h;

    // Draw original grayscale mask to get pixel data
    ctx.drawImage(rawMaskImage, 0, 0, w, h);
    const imageData = ctx.getImageData(0, 0, w, h);
    const data = imageData.data;

    // Threshold value (0-255)
    const threshVal = threshold * 255;

    // Loop through pixels
    for (let i = 0; i < data.length; i += 4) {
      const gray = data[i]; // R channel (since it's grayscale, R=G=B)
      
      if (gray < threshVal) {
        // Below threshold: Make Transparent
        data[i + 3] = 0; 
      } else {
        // Above threshold: Colorize (Teal)
        // Set Alpha based on intensity, but boost it for visibility
        data[i] = 0;     // R
        data[i+1] = 255; // G
        data[i+2] = 220; // B
        data[i+3] = 180; // Fixed alpha for visibility, or use `gray` for variable
      }
    }

    ctx.putImageData(imageData, 0, 0);

  }, [rawMaskImage, threshold]);


  // --- RENDER HELPERS ---

  return (
    <div className="flex flex-col h-screen bg-neutral-900 text-neutral-100 font-sans overflow-hidden">
      
      {/* HEADER */}
      <header className="h-14 border-b border-neutral-700 flex items-center justify-between px-4 bg-neutral-800 shrink-0 z-10">
        <div className="flex items-center gap-2">
          <Layers className="text-teal-400" />
          <h1 className="font-bold text-lg tracking-tight">SAM3 GeoLabeler</h1>
        </div>
        
        <div className="flex items-center gap-4">
          {appState === 'labeling' && (
             <div className="flex items-center bg-neutral-900 rounded-lg p-1 border border-neutral-700">
                <button 
                  onClick={() => setTool('pan')}
                  className={`p-2 rounded ${tool === 'pan' ? 'bg-neutral-700 text-white' : 'text-neutral-400 hover:text-white'}`}
                  title="Pan Tool (Space+Drag)"
                >
                  <Hand size={18} />
                </button>
                <button 
                  onClick={() => setTool('box')}
                  className={`p-2 rounded ${tool === 'box' ? 'bg-neutral-700 text-white' : 'text-neutral-400 hover:text-white'}`}
                  title="Box Tool"
                >
                  <BoxSelect size={18} />
                </button>
                <div className="w-px h-6 bg-neutral-700 mx-1"></div>
                <button onClick={() => setScale(s => s * 1.2)} className="p-2 text-neutral-400 hover:text-white"><ZoomIn size={18} /></button>
                <button onClick={() => setScale(s => s / 1.2)} className="p-2 text-neutral-400 hover:text-white"><ZoomOut size={18} /></button>
             </div>
          )}
           <button 
            onClick={handleReset} 
            className="text-xs flex items-center gap-1 text-red-400 hover:text-red-300 border border-red-900/50 hover:border-red-500/50 bg-red-900/10 px-3 py-1.5 rounded-md transition-colors"
          >
            <Trash2 size={14} /> Reset Session
          </button>
        </div>
      </header>

      {/* MAIN CONTENT */}
      <div className="flex flex-1 overflow-hidden relative">
        
        {/* LEFT SIDEBAR (Controls) */}
        {appState === 'labeling' && (
          <aside className="w-80 bg-neutral-800 border-r border-neutral-700 flex flex-col p-4 gap-6 z-10 shadow-xl overflow-y-auto">
            
            {/* Box Management */}
            <div>
              <h3 className="text-xs font-semibold uppercase text-neutral-500 mb-3 tracking-wider">Detection</h3>
              <div className="bg-neutral-900 rounded-lg p-3 border border-neutral-700">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-neutral-300">Input Boxes: {boxes.length}</span>
                  <button 
                    onClick={() => setBoxes([])}
                    className="text-xs text-red-400 hover:text-red-300"
                  >
                    Clear All
                  </button>
                </div>
                <button
                  onClick={handleSegment}
                  disabled={loading || boxes.length === 0}
                  className="w-full mt-2 py-2 bg-teal-600 hover:bg-teal-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium rounded flex items-center justify-center gap-2 transition-colors"
                >
                  {loading && isSegmenting ? <RefreshCw className="animate-spin" size={16}/> : <MousePointer2 size={16}/>}
                  Segment Objects
                </button>
              </div>
            </div>

            {/* Threshold & Results */}
            <div className={`transition-opacity duration-300 ${rawMaskImage ? 'opacity-100' : 'opacity-40 pointer-events-none'}`}>
              <h3 className="text-xs font-semibold uppercase text-neutral-500 mb-3 tracking-wider">Refinement</h3>
              
              <div className="bg-neutral-900 rounded-lg p-3 border border-neutral-700">
                <div className="flex justify-between mb-2">
                  <label className="text-sm text-neutral-300">Confidence Threshold</label>
                  <span className="text-sm font-mono text-teal-400">{threshold.toFixed(2)}</span>
                </div>
                <input 
                  type="range" 
                  min="0" max="1" step="0.01" 
                  value={threshold} 
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-teal-500"
                />
                <p className="text-xs text-neutral-500 mt-2">
                  Adjust slider to filter noise. Results below this value will be excluded from export.
                </p>
              </div>
            </div>

            {/* Export */}
            <div className={`mt-auto transition-opacity duration-300 ${rawMaskImage ? 'opacity-100' : 'opacity-40 pointer-events-none'}`}>
              <button
                onClick={handleDownloadGeoData}
                disabled={loading}
                className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 text-white font-medium rounded flex items-center justify-center gap-2 shadow-lg transition-colors"
              >
                {loading && !isSegmenting ? <RefreshCw className="animate-spin" size={18}/> : <Download size={18}/>}
                Download GeoData (.zip)
              </button>
            </div>
          </aside>
        )}

        {/* CENTER STAGE */}
        <div className="flex-1 bg-neutral-950 relative overflow-hidden cursor-crosshair">
          
          {appState === 'input' ? (
            // INPUT SCREEN
            <div className="absolute inset-0 flex items-center justify-center bg-neutral-900 z-50">
              <div className="w-full max-w-md p-8 bg-neutral-800 rounded-xl shadow-2xl border border-neutral-700">
                <div className="flex justify-center mb-6">
                  <div className="p-4 bg-teal-500/10 rounded-full">
                    <Upload className="text-teal-400" size={48} />
                  </div>
                </div>
                <h2 className="text-2xl font-bold text-center mb-2">Load Geospatial Imagery</h2>
                <p className="text-center text-neutral-400 mb-6 text-sm">Enter a URL to a satellite image, map, or photo to begin segmentation.</p>
                
                <div className="space-y-4">
                  <input
                    type="text"
                    placeholder="https://example.com/image.png"
                    value={imageUrl}
                    onChange={(e) => setImageUrl(e.target.value)}
                    className="w-full p-3 bg-neutral-900 border border-neutral-700 rounded-lg focus:ring-2 focus:ring-teal-500 outline-none transition-all"
                  />
                  <button
                    onClick={handleUpload}
                    disabled={loading || !imageUrl}
                    className="w-full py-3 bg-teal-600 hover:bg-teal-500 disabled:opacity-50 text-white font-bold rounded-lg transition-colors flex justify-center items-center gap-2"
                  >
                    {loading ? "Uploading..." : "Load Image"}
                  </button>
                  {error && (
                    <div className="p-3 bg-red-900/20 border border-red-900/50 rounded text-red-300 text-sm flex items-center gap-2">
                       <AlertCircle size={16} /> {error}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            // CANVAS VIEWER
            <div 
              ref={containerRef}
              className="w-full h-full relative overflow-hidden select-none"
              onMouseDown={onMouseDown}
              onMouseMove={onMouseMove}
              onMouseUp={onMouseUp}
              onMouseLeave={onMouseUp}
              onWheel={onWheel}
              style={{ cursor: tool === 'pan' || isDragging ? 'grab' : 'crosshair' }}
            >
              {/* TRANSFORM CONTAINER */}
              <div 
                style={{
                  transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
                  transformOrigin: '0 0',
                  width: imageDimensions.width,
                  height: imageDimensions.height,
                  position: 'absolute'
                }}
              >
                {/* 1. Base Image */}
                {displayImageSrc && (
                  <img 
                    src={displayImageSrc} 
                    alt="Workplace" 
                    className="absolute top-0 left-0 pointer-events-none"
                    onLoad={handleImageLoad}
                    draggable={false}
                  />
                )}

                {/* 2. Mask Overlay (Canvas) */}
                <canvas 
                  ref={maskCanvasRef}
                  className="absolute top-0 left-0 pointer-events-none mix-blend-screen opacity-90"
                  style={{ width: imageDimensions.width, height: imageDimensions.height }}
                />

                {/* 3. Box Drawing Layer (SVG) */}
                <svg className="absolute top-0 left-0 w-full h-full pointer-events-none">
                  {/* Existing Boxes */}
                  {boxes.map((box, i) => (
                    <rect
                      key={i}
                      x={box[0]}
                      y={box[1]}
                      width={box[2] - box[0]}
                      height={box[3] - box[1]}
                      fill="rgba(0,0,0,0)"
                      stroke="#4fd1c5" // Teal-400
                      strokeWidth={2 / scale}
                    />
                  ))}
                  
                  {/* Box currently being drawn */}
                  {currentBox && (
                    <rect
                      x={Math.min(currentBox.x1, currentBox.x2)}
                      y={Math.min(currentBox.y1, currentBox.y2)}
                      width={Math.abs(currentBox.x2 - currentBox.x1)}
                      height={Math.abs(currentBox.y2 - currentBox.y1)}
                      fill="rgba(79, 209, 197, 0.2)"
                      stroke="#ffffff"
                      strokeWidth={2 / scale}
                      strokeDasharray={`${5/scale}`}
                    />
                  )}
                </svg>
              </div>

              {/* OVERLAY LOADING SPINNER */}
              {loading && (
                <div className="absolute top-4 right-4 bg-neutral-900/80 backdrop-blur text-white px-4 py-2 rounded-full flex items-center gap-2 shadow-lg border border-neutral-700 z-50">
                  <RefreshCw className="animate-spin" size={16} /> Processing...
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
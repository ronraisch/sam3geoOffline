"use client"

import type React from "react"

import { useState, useRef, useCallback, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Slider } from "@/components/ui/slider"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Loader2, ZoomIn, ZoomOut, Move, Square, Download, Trash2 } from "lucide-react"

const API_BASE = "https://bureau-originally-prayer-describe.trycloudflare.com"

interface BoundingBox {
  id: string
  x1: number
  y1: number
  x2: number
  y2: number
}

type Tool = "pan" | "box"

export function SegmentationTool() {
  const [imageUrl, setImageUrl] = useState("")
  const [internalPath, setInternalPath] = useState<string | null>(null)
  const [imageLoaded, setImageLoaded] = useState(false)
  const [loading, setLoading] = useState(false)
  const [segmenting, setSegmenting] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Canvas state
  const [scale, setScale] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [boxes, setBoxes] = useState<BoundingBox[]>([])
  const [currentTool, setCurrentTool] = useState<Tool>("box")
  const [isDrawing, setIsDrawing] = useState(false)
  const [isPanning, setIsPanning] = useState(false)
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null)
  const [panStart, setPanStart] = useState<{ x: number; y: number } | null>(null)
  const [currentBox, setCurrentBox] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null)

  // Mask state
  const [maskPath, setMaskPath] = useState<string | null>(null)
  const [threshold, setThreshold] = useState(0.5)

  // Refs
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const maskImageRef = useRef<HTMLImageElement | null>(null)

  // Load image from API
  const handleLoadImage = async () => {
    if (!imageUrl.trim()) return

    setLoading(true)
    setError(null)
    setBoxes([])
    setMaskPath(null)
    setInternalPath(null)
    setImageLoaded(false)

    try {
      const response = await fetch(`${API_BASE}/upload-path`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: imageUrl }),
      })

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || "Failed to upload image")
      }

      const data = await response.json()
      setInternalPath(data.internal_path)

      // Load the image for display
      const img = new Image()
      img.crossOrigin = "anonymous"
      img.onload = () => {
        imageRef.current = img
        setImageLoaded(true)
        setScale(1)
        setOffset({ x: 0, y: 0 })
        setLoading(false)
      }
      img.onerror = () => {
        setError("Failed to load image for display")
        setLoading(false)
      }
      img.src = `${API_BASE}/download/${data.internal_path}`
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error")
      setLoading(false)
    }
  }

  // Segment with boxes
  const handleSegment = async () => {
    if (boxes.length === 0) {
      setError("Please draw at least one bounding box")
      return
    }

    setSegmenting(true)
    setError(null)

    try {
      const boxesData = boxes.map((b) => [
        Math.min(b.x1, b.x2),
        Math.min(b.y1, b.y2),
        Math.max(b.x1, b.x2),
        Math.max(b.y1, b.y2),
      ])

      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 20000)

      const response = await fetch(`${API_BASE}/predict-boxes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ boxes: boxesData }),
        signal: controller.signal,
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || "Failed to segment")
      }

      const data = await response.json()
      setMaskPath(data.sum_image_url)

      // Load mask image
      const maskImg = new Image()
      maskImg.crossOrigin = "anonymous"
      maskImg.onload = () => {
        maskImageRef.current = maskImg
        setSegmenting(false)
      }
      maskImg.onerror = () => {
        setError("Failed to load mask image")
        setSegmenting(false)
      }
      maskImg.src = `${API_BASE}/download/${data.sum_image_url}`
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error")
      setSegmenting(false)
    }
  }

  // Export geodata
  const handleExport = async () => {
    setExporting(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/export-geodata?threshold=${threshold}`)

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || err.message || "Failed to export")
      }

      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `export_${threshold}.zip`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error")
    } finally {
      setExporting(false)
    }
  }

  // Get canvas coordinates from mouse event
  const getCanvasCoords = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current
      if (!canvas) return { x: 0, y: 0 }

      const rect = canvas.getBoundingClientRect()
      const x = (e.clientX - rect.left - offset.x) / scale
      const y = (e.clientY - rect.top - offset.y) / scale
      return { x, y }
    },
    [scale, offset],
  )

  // Mouse handlers
  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!imageLoaded) return

      const coords = getCanvasCoords(e)

      if (currentTool === "pan" || e.button === 1) {
        setIsPanning(true)
        setPanStart({ x: e.clientX - offset.x, y: e.clientY - offset.y })
      } else if (currentTool === "box") {
        setIsDrawing(true)
        setDrawStart(coords)
        setCurrentBox({ x1: coords.x, y1: coords.y, x2: coords.x, y2: coords.y })
      }
    },
    [imageLoaded, currentTool, getCanvasCoords, offset],
  )

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (isPanning && panStart) {
        setOffset({
          x: e.clientX - panStart.x,
          y: e.clientY - panStart.y,
        })
      } else if (isDrawing && drawStart) {
        const coords = getCanvasCoords(e)
        setCurrentBox({
          x1: drawStart.x,
          y1: drawStart.y,
          x2: coords.x,
          y2: coords.y,
        })
      }
    },
    [isPanning, panStart, isDrawing, drawStart, getCanvasCoords],
  )

  const handleMouseUp = useCallback(() => {
    if (isDrawing && currentBox) {
      const width = Math.abs(currentBox.x2 - currentBox.x1)
      const height = Math.abs(currentBox.y2 - currentBox.y1)

      if (width > 5 && height > 5) {
        setBoxes((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            ...currentBox,
          },
        ])
      }
    }

    setIsDrawing(false)
    setIsPanning(false)
    setDrawStart(null)
    setPanStart(null)
    setCurrentBox(null)
  }, [isDrawing, currentBox])

  // Zoom handlers
  const handleZoomIn = () => setScale((s) => Math.min(s * 1.2, 5))
  const handleZoomOut = () => setScale((s) => Math.max(s / 1.2, 0.1))

  const handleWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? 0.9 : 1.1
    setScale((s) => Math.min(Math.max(s * delta, 0.1), 5))
  }, [])

  // Clear boxes
  const handleClearBoxes = () => {
    setBoxes([])
    setMaskPath(null)
    maskImageRef.current = null
  }

  // Draw canvas
  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext("2d")
    if (!canvas || !ctx) return

    const container = containerRef.current
    if (container) {
      canvas.width = container.clientWidth
      canvas.height = container.clientHeight
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw background
    ctx.fillStyle = "#1a1a2e"
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    if (!imageRef.current || !imageLoaded) {
      ctx.fillStyle = "#888"
      ctx.font = "16px sans-serif"
      ctx.textAlign = "center"
      ctx.fillText("Enter an image URL and click Load", canvas.width / 2, canvas.height / 2)
      return
    }

    ctx.save()
    ctx.translate(offset.x, offset.y)
    ctx.scale(scale, scale)

    // Draw image
    ctx.drawImage(imageRef.current, 0, 0)

    // Draw mask with threshold
    if (maskImageRef.current && maskPath) {
      const maskCanvas = document.createElement("canvas")
      maskCanvas.width = imageRef.current.width
      maskCanvas.height = imageRef.current.height
      const maskCtx = maskCanvas.getContext("2d")!

      maskCtx.drawImage(maskImageRef.current, 0, 0)
      const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height)
      const data = maskData.data

      // Apply threshold and colorize
      for (let i = 0; i < data.length; i += 4) {
        const value = data[i] / 255
        if (value >= threshold) {
          data[i] = 0 // R
          data[i + 1] = 200 // G
          data[i + 2] = 100 // B
          data[i + 3] = 150 // A
        } else {
          data[i + 3] = 0 // Transparent
        }
      }

      maskCtx.putImageData(maskData, 0, 0)
      ctx.drawImage(maskCanvas, 0, 0)
    }

    // Draw existing boxes
    ctx.strokeStyle = "#00ff00"
    ctx.lineWidth = 2 / scale
    boxes.forEach((box) => {
      ctx.strokeRect(
        Math.min(box.x1, box.x2),
        Math.min(box.y1, box.y2),
        Math.abs(box.x2 - box.x1),
        Math.abs(box.y2 - box.y1),
      )
    })

    // Draw current box
    if (currentBox) {
      ctx.strokeStyle = "#ffff00"
      ctx.setLineDash([5 / scale, 5 / scale])
      ctx.strokeRect(
        Math.min(currentBox.x1, currentBox.x2),
        Math.min(currentBox.y1, currentBox.y2),
        Math.abs(currentBox.x2 - currentBox.x1),
        Math.abs(currentBox.y2 - currentBox.y1),
      )
      ctx.setLineDash([])
    }

    ctx.restore()
  }, [imageLoaded, scale, offset, boxes, currentBox, maskPath, threshold])

  // Resize handler
  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current
      const container = containerRef.current
      if (canvas && container) {
        canvas.width = container.clientWidth
        canvas.height = container.clientHeight
      }
    }

    window.addEventListener("resize", handleResize)
    handleResize()
    return () => window.removeEventListener("resize", handleResize)
  }, [])

  return (
    <div className="flex flex-col lg:flex-row h-screen">
      {/* Sidebar */}
      <div className="w-full lg:w-80 p-4 border-b lg:border-b-0 lg:border-r border-border bg-card overflow-y-auto">
        <h1 className="text-xl font-bold mb-4 text-foreground">SAM Segmentation Tool</h1>

        {/* Image URL Input */}
        <Card className="mb-4">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Load Image</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <Input
              placeholder="Enter image URL or path..."
              value={imageUrl}
              onChange={(e) => setImageUrl(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleLoadImage()}
            />
            <Button onClick={handleLoadImage} disabled={loading || !imageUrl.trim()} className="w-full">
              {loading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
              Load Image
            </Button>
          </CardContent>
        </Card>

        {/* Tools */}
        {imageLoaded && (
          <>
            <Card className="mb-4">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Tools</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex gap-2 flex-wrap">
                  <Button
                    variant={currentTool === "box" ? "default" : "outline"}
                    size="sm"
                    onClick={() => setCurrentTool("box")}
                  >
                    <Square className="h-4 w-4 mr-1" /> Draw Box
                  </Button>
                  <Button
                    variant={currentTool === "pan" ? "default" : "outline"}
                    size="sm"
                    onClick={() => setCurrentTool("pan")}
                  >
                    <Move className="h-4 w-4 mr-1" /> Pan
                  </Button>
                  <Button variant="outline" size="sm" onClick={handleZoomIn}>
                    <ZoomIn className="h-4 w-4" />
                  </Button>
                  <Button variant="outline" size="sm" onClick={handleZoomOut}>
                    <ZoomOut className="h-4 w-4" />
                  </Button>
                </div>
                <div className="mt-2 text-xs text-muted-foreground">
                  Boxes: {boxes.length} | Zoom: {Math.round(scale * 100)}%
                </div>
                {boxes.length > 0 && (
                  <Button variant="ghost" size="sm" onClick={handleClearBoxes} className="mt-2 text-destructive">
                    <Trash2 className="h-4 w-4 mr-1" /> Clear Boxes
                  </Button>
                )}
              </CardContent>
            </Card>

            {/* Segment Button */}
            <Card className="mb-4">
              <CardContent className="pt-4">
                <Button onClick={handleSegment} disabled={segmenting || boxes.length === 0} className="w-full">
                  {segmenting ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                  Segment
                </Button>
              </CardContent>
            </Card>

            {/* Threshold Slider - only show after segmentation */}
            {maskPath && (
              <Card className="mb-4">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Threshold</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="flex items-center gap-4">
                    <Slider
                      value={[threshold]}
                      onValueChange={([v]) => setThreshold(v)}
                      min={0}
                      max={1}
                      step={0.01}
                      className="flex-1"
                    />
                    <span className="text-sm font-mono w-12 text-foreground">{threshold.toFixed(2)}</span>
                  </div>
                  <Label className="text-xs text-muted-foreground">Adjust to filter mask by confidence</Label>
                </CardContent>
              </Card>
            )}

            {/* Export Button */}
            {maskPath && (
              <Card>
                <CardContent className="pt-4">
                  <Button onClick={handleExport} disabled={exporting} variant="secondary" className="w-full">
                    {exporting ? (
                      <Loader2 className="h-4 w-4 animate-spin mr-2" />
                    ) : (
                      <Download className="h-4 w-4 mr-2" />
                    )}
                    Download Layer
                  </Button>
                </CardContent>
              </Card>
            )}
          </>
        )}

        {/* Error Display */}
        {error && (
          <div className="mt-4 p-3 bg-destructive/10 border border-destructive/20 rounded-md text-sm text-destructive">
            {error}
          </div>
        )}
      </div>

      {/* Canvas Area */}
      <div ref={containerRef} className="flex-1 relative bg-muted">
        <canvas
          ref={canvasRef}
          className={`w-full h-full ${currentTool === "pan" ? "cursor-grab" : "cursor-crosshair"} ${isPanning ? "cursor-grabbing" : ""}`}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onWheel={handleWheel}
          onContextMenu={(e) => e.preventDefault()}
        />
      </div>
    </div>
  )
}

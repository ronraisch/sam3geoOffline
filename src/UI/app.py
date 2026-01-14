import asyncio
from typing import List, Optional, Tuple
import httpx
from nicegui import ui, events

# --- Configuration ---
# Update this if your FastAPI server is on a different port or host
API_BASE_URL = "http://localhost:8000"

class AppState:
    def __init__(self):
        self.display_image_url: Optional[str] = None
        self.mask_url: Optional[str] = None
        self.boxes: List[List[float]] = []
        self.temp_box: Optional[List[float]] = None
        self.is_drawing: bool = False
        self.start_pos: Tuple[float, float] = (0, 0)
        self.text_prompt: str = ""
        self.mask_opacity: float = 0.5
        self.image_link_input: str = ""

state = AppState()

# --- API Interaction ---

async def load_image_from_link():
    """Sends a link to /upload-path and gets the internal path for the download route."""
    if not state.image_link_input:
        ui.notify("Please enter an image URL", type="warning")
        return

    async with httpx.AsyncClient() as client:
        try:
            ui.notify("Uploading image link to server...", spinner=True)
            response = await client.post(
                f"{API_BASE_URL}/upload-path",
                json={"path": state.image_link_input},
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
            # The API returns 'internal_path' like 'uploads/unique_id.png'
            # We use the API's download route to show it in the UI
            internal_path = data["internal_path"]
            state.display_image_url = f"{API_BASE_URL}/download/{internal_path}"
            
            # Reset state for new image
            state.boxes = []
            state.mask_url = None
            update_ui()
            ui.notify("Image loaded successfully", type="positive")
            
        except Exception as e:
            ui.notify(f"Error loading image: {str(e)}", type="negative")
            print(e)

async def run_segmentation():
    """Calls /predict-mask and sets the mask URL."""
    if not state.display_image_url:
        ui.notify("No image loaded on server.", type="warning")
        return

    payload = {
        "boxes": state.boxes if state.boxes else None,
        "text": state.text_prompt if state.text_prompt else None
    }
    
    if not payload["boxes"] and not payload["text"]:
        ui.notify("Provide a box or text prompt.", type="warning")
        return

    async with httpx.AsyncClient() as client:
        try:
            ui.notify("Processing SAM3...", spinner=True)
            response = await client.post(
                f"{API_BASE_URL}/predict-mask", 
                json=payload,
                timeout=90.0
            )
            response.raise_for_status()
            data = response.json()
            
            # Construct the download link for the generated mask
            state.mask_url = f"{API_BASE_URL}/download/{data['sum_image_url']}"
            update_ui()
            ui.notify(f"Found {data['mask_count']} masks", type="positive")
            
        except Exception as e:
            ui.notify(f"Segmentation failed: {str(e)}", type="negative")

# --- UI Logic & Drawing ---

def handle_mouse(e: events.MouseEventArguments):
    if not state.display_image_url:
        return
        
    if e.type == 'mousedown':
        state.is_drawing = True
        state.start_pos = (e.image_x, e.image_y)
        
    elif e.type == 'mousemove' and state.is_drawing:
        state.temp_box = [
            min(state.start_pos[0], e.image_x), min(state.start_pos[1], e.image_y),
            max(state.start_pos[0], e.image_x), max(state.start_pos[1], e.image_y)
        ]
        update_svg()
            
    elif e.type == 'mouseup' and state.is_drawing:
        state.is_drawing = False
        box = [
            min(state.start_pos[0], e.image_x), min(state.start_pos[1], e.image_y),
            max(state.start_pos[0], e.image_x), max(state.start_pos[1], e.image_y)
        ]
        if (box[2] - box[0]) > 2 and (box[3] - box[1]) > 2:
            state.boxes.append(box)
        state.temp_box = None
        update_svg()

def update_svg():
    content = ""
    # Overlay the mask from the API
    if state.mask_url:
        content += f'<image href="{state.mask_url}" x="0" y="0" width="100%" height="100%" opacity="{state.mask_opacity}" />'

    # Draw existing boxes
    for b in state.boxes:
        content += f'<rect x="{b[0]}" y="{b[1]}" width="{b[2]-b[0]}" height="{b[3]-b[1]}" fill="none" stroke="#00ff00" stroke-width="2" />'

    # Draw active drag box
    if state.temp_box:
        b = state.temp_box
        content += f'<rect x="{b[0]}" y="{b[1]}" width="{b[2]-b[0]}" height="{b[3]-b[1]}" fill="rgba(0,255,0,0.2)" stroke="#00ff00" stroke-width="1" stroke-dasharray="4" />'

    img_viewer.content = content

def update_ui():
    if state.display_image_url:
        img_viewer.set_source(state.display_image_url)
    update_svg()

# --- Layout ---

with ui.row().classes('w-full h-screen no-wrap'):
    
    # Sidebar
    with ui.card().classes('w-96 h-full p-4 bg-slate-50'):
        ui.label('SAM3 Interface').classes('text-2xl font-bold mb-4')
        
        # URL Input
        ui.label('Step 1: Load Image').classes('font-bold mt-2')
        url_input = ui.input('Image URL or Local Path', 
                             on_change=lambda e: setattr(state, 'image_link_input', e.value)) \
                             .classes('w-full')
        ui.button('Load Image', on_click=load_image_from_link).classes('w-full mb-4')
        
        ui.separator()
        
        # Bounding Box Controls
        ui.label('Step 2: Prompts').classes('font-bold mt-2')
        with ui.row().classes('w-full gap-2'):
            ui.button('Undo Box', icon='undo', 
                      on_click=lambda: (state.boxes.pop() if state.boxes else None, update_svg())) \
                      .props('outline size=sm').classes('flex-1')
            ui.button('Clear Boxes', icon='delete', 
                      on_click=lambda: (setattr(state, 'boxes', []), update_svg())) \
                      .props('outline color=red size=sm').classes('flex-1')
        
        # Text Prompt
        ui.input('Text Prompt', placeholder='e.g. "car"', 
                 on_change=lambda e: setattr(state, 'text_prompt', e.value)).classes('w-full')
        
        # Segment Button
        ui.button('SEGMENT', on_click=run_segmentation).classes('w-full bg-blue-700 text-white font-bold h-12 mt-4')
        
        ui.separator()
        
        # Mask visualization
        ui.label('Step 3: Visualize').classes('font-bold mt-2')
        ui.label('Mask Opacity')
        ui.slider(min=0, max=1, step=0.05, value=0.5, 
                  on_change=lambda e: (setattr(state, 'mask_opacity', e.value), update_svg())).props('label-always')

    # Main Viewer
    with ui.column().classes('flex-grow h-full items-center justify-center p-4 bg-gray-200'):
        img_viewer = ui.interactive_image(
            on_mouse=handle_mouse,
            events=['mousedown', 'mouseup', 'mousemove'],
            cross=True
        ).classes('max-h-full shadow-2xl bg-white')
        
        # Initial placeholder
        img_viewer.set_source('https://placehold.co/600x400?text=Paste+URL+to+Start')

ui.run(title="SAM3 Geospatial UI", port=8080)
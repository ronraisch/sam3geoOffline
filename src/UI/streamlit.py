import streamlit as st
import rasterio
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from PIL import Image
import numpy as np
import io
import base64
import time


# --- Logging Utility ---
def log_event(message):
    """Logs events to terminal and a session state list for UI display."""
    timestamp = time.strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    print(full_msg)
    if "logs" not in st.session_state:
        st.session_state.logs = []
    st.session_state.logs.append(full_msg)
    st.session_state.logs = st.session_state.logs[-20:]


@st.cache_resource
def load_geotiff_data(path):
    """
    Detailed loading to handle the GCP-based transform.
    Returns the base64 URL and bounds for Folium.
    """
    log_event(f"CACHE MISS: Loading GeoTIFF from {path}")
    try:
        with rasterio.open(path) as src:
            height, width = src.height, src.width
            tl = src.transform * (0, 0)
            tr = src.transform * (width, 0)
            br = src.transform * (width, height)
            bl = src.transform * (0, height)

            lons = [tl[0], tr[0], br[0], bl[0]]
            lats = [tl[1], tr[1], br[1], bl[1]]
            folium_bounds = [[min(lons), min(lats)], [max(lons), max(lats)]]

            img_data = src.read([1, 2, 3])
            img_data = np.transpose(img_data, (1, 2, 0))

            if img_data.dtype != np.uint8:
                img_data = (
                    (img_data - img_data.min())
                    / (img_data.max() - img_data.min())
                    * 255
                ).astype(np.uint8)

            buffered = io.BytesIO()
            Image.fromarray(img_data).save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            img_url = f"data:image/png;base64,{img_str}"

            return img_url, folium_bounds, src.crs
    except Exception as e:
        log_event(f"FILE ERROR: {str(e)}")
        return None, None, None


def main():
    st.set_page_config(layout="wide", page_title="GeoTIFF Digitizer")

    # 1. Initialize Session State
    if "captured_drawings" not in st.session_state:
        st.session_state.captured_drawings = []
    if "logs" not in st.session_state:
        st.session_state.logs = []

    st.title("GeoTIFF Digitizer")

    with st.sidebar:
        st.header("Settings & Logs")
        if st.button("Clear Drawings"):
            st.session_state.captured_drawings = []
            st.rerun()
        if st.button("Reset All Cache"):
            st.cache_resource.clear()
            st.session_state.captured_drawings = []
            st.rerun()

        st.markdown("---")
        for log in reversed(st.session_state.get("logs", [])):
            st.text(log)

    tiff_path = "data/example/netivot1.tif"
    img_url, bounds, crs = load_geotiff_data(tiff_path)

    if img_url:
        # 2. Re-create the Map Object on every rerun
        center = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
        m = folium.Map(location=center, zoom_start=18)

        # Add the GeoTIFF
        folium.raster_layers.ImageOverlay(
            image=img_url, bounds=bounds, opacity=1, name="GeoTIFF"
        ).add_to(m)

        # 3. ALTERNATIVE SHADOW FIX:
        # Since Draw(data=...) is not available in your version, we add existing
        # drawings as a non-interactive GeoJSON layer. This stops the "shadowing"
        # behavior where the drawing tool tries to interact with existing shapes
        # that it didn't create in the current session.
        if st.session_state.captured_drawings:
            for drawing in st.session_state.captured_drawings:
                folium.GeoJson(
                    drawing,
                    interactive=False,  # Crucial to prevent shadowing/interaction conflicts
                    style_function=lambda x: {
                        "fillColor": "#3388ff",
                        "color": "#3388ff",
                    },
                ).add_to(m)

        # Add the Draw control
        Draw(
            export=False,
            draw_options={
                "polyline": False,
                "circle": False,
                "marker": False,
                "circlemarker": False,
                "polygon": True,
                "rectangle": True,
            },
        ).add_to(m)

        col1, col2 = st.columns([4, 1])

        with col1:
            # 4. Use st_folium with the dynamic map
            output = st_folium(
                m,
                width="100%",
                height=600,
                key="geo_map",
                returned_objects=["all_drawings"],
                use_container_width=True,
            )

        with col2:
            st.info(f"CRS: {crs}")

            # 5. Check for new drawings and update state
            if output and output.get("all_drawings") is not None:
                new_drawings = output["all_drawings"]
                # Only rerun if the data actually changed
                if new_drawings != st.session_state.captured_drawings:
                    st.session_state.captured_drawings = new_drawings
                    log_event(f"Captured {len(new_drawings)} drawing(s)")
                    st.rerun()

            st.write(f"Total polygons: {len(st.session_state.captured_drawings)}")
    else:
        st.error("Could not load GeoTIFF. Check the file path.")


if __name__ == "__main__":
    main()

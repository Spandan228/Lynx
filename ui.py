"""
Nexus Enterprise Dashboard & Agentic CRAG Interface (Streamlit Workspace)
Direct 1:1 Replica of sample.mp4

Embedded full-fidelity macOS dashboard shell with:
- macOS Window Chrome (Traffic lights, Omnibox 'nexus.io')
- Left Sidebar (Nexus Ribbon Logo, Categorized Navigation, Team Switcher)
- Top Bar (Universal Search '⌘ + F', User Avatar)
- Top 3 KPI Bento Cards (Page Views 12,450 ↗, Total Revenue $363.95 ↘, Bounce Rate 86.5% ↗)
- Sales Overview Flow Stream Graph & Weekly Subscriber Bar Chart
- Sales Distribution Donut Gauge & List of Integration Data Table
- Sliding Agentic CRAG Copilot with Real-Time SSE Token Streaming (/stream_query)
"""

import os
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Page Configuration: Fullscreen Layout
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Lynx CRAG • Enterprise Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS Injection: Remove Default Streamlit Chrome & Margins
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Hide Streamlit Header, Footer, and Menus */
    #MainMenu {visibility: hidden; display: none !important;}
    header {visibility: hidden; display: none !important;}
    footer {visibility: hidden; display: none !important;}
    [data-testid="stSidebar"] {display: none !important;}
    [data-testid="collapsedControl"] {display: none !important;}
    
    /* Zero Margins & Full Viewport Expansion */
    .block-container {
        padding: 0rem !important;
        margin: 0rem !important;
        max-width: 100% !important;
    }
    
    /* Target Component iframe */
    iframe {
        width: 100% !important;
        border: none !important;
        display: block !important;
        min-height: 98vh !important;
    }
    
    body {
        background-color: #090d16 !important;
        overflow: hidden !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_bundled_ui() -> str:
    """Reads static HTML, CSS, and JS and inlines them into a single self-contained application."""
    static_dir = Path(__file__).parent / "static"
    html_path = static_dir / "index.html"
    css_path = static_dir / "styles.css"
    js_path = static_dir / "app.js"

    html_content = html_path.read_text(encoding="utf-8")
    css_content = css_path.read_text(encoding="utf-8")
    js_content = js_path.read_text(encoding="utf-8")

    # Inline CSS into HTML <head>
    inlined_css = f"<style>\n{css_content}\n</style>"
    html_content = html_content.replace('<link rel="stylesheet" href="styles.css">', inlined_css)

    # Inline JS before </body>
    inlined_js = f"<script>\n{js_content}\n</script>"
    html_content = html_content.replace('<script src="app.js"></script>', inlined_js)

    return html_content


# ---------------------------------------------------------------------------
# Render Full-Fidelity Nexus Dashboard
# ---------------------------------------------------------------------------
bundled_html = load_bundled_ui()
components.html(bundled_html, height=1250, scrolling=False)

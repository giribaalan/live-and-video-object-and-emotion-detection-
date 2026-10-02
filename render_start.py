"""Production entry point used by Render's Gunicorn service."""

from app import app, load_config

# Keep the local dashboard settings when present.  On Render, configure an
# accessible video/RTSP source through the dashboard after deployment.
load_config()

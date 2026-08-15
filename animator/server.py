"""Tiny local static file server so model-viewer can fetch the .glb (browsers block
binary fetch()/XHR against file:// URLs, so we can't just open viewer.html directly)."""
from __future__ import annotations

import functools
import http.server
import threading


def serve_directory(directory: str, port: int = 0) -> str:
    """Start a background HTTP server rooted at `directory`. Returns its base URL."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{httpd.server_port}"

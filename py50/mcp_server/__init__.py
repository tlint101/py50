"""
Setup for MCP server for exposing py50's dose-response calculations and plots.
Need to install optional dependencies with ``pip install 'py50[mcp]'``
Server is launched with the ``py50-mcp`` console script.
"""

import os

# keep matplotlib image from displaying to avoid MCP server issues
os.environ.setdefault("MPLBACKEND", "Agg")

from py50.mcp_server.server import main, mcp

__all__ = ["main", "mcp"]

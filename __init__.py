"""Top-level package for pipazoul_utils."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """pipazoul-comfyui-utils"""
__email__ = "yassin@siouda.com"
__version__ = "0.0.1"

from .src.pipazoul_utils.nodes import NODE_CLASS_MAPPINGS
from .src.pipazoul_utils.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

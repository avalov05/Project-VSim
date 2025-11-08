"""
Virtual In Silico Virus Laboratory
Comprehensive computational virology platform
"""

__version__ = "2.0.0"
__author__ = "VSim Lab"

from .core.pipeline import VLabPipeline
from .core.config import VLabConfig

__all__ = ['VLabPipeline', 'VLabConfig']


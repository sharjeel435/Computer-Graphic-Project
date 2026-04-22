"""
Time-Reversal Physics Simulator - 3D Graphics Application
A sophisticated computer graphics project demonstrating real-time physics
simulation with complete time-reversal capabilities using PyOpenGL.
"""

__version__ = "1.0.0"
__author__ = "CG Project Team"
__all__ = ["main"]

try:
    import numpy as np
    import pygame
    from OpenGL.GL import *
except ImportError as e:
    print(f"❌ Missing required dependency: {e}")
    print("Install with: pip install -r requirements.txt")
    raise

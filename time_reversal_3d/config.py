"""
Configuration and Constants for Time-Reversal Physics Simulator
"""

import numpy as np
from enum import Enum

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900
FPS = 60
BACKGROUND_COLOR = (0.05, 0.05, 0.1)  # Dark blue
WINDOW_TITLE = "Time-Reversal Physics Simulator"

# ============================================================================
# PHYSICS SETTINGS
# ============================================================================

GRAVITY = 9.81  # m/s^2
PHYSICS_TIMESTEP = 1.0 / 60.0  # Fixed timestep
PHYSICS_SUBSTEPS = 1

# Object Settings
OBJECT_COUNT = 40
OBJECT_SIZE_MIN = 0.3
OBJECT_SIZE_MAX = 0.8
OBJECT_MASS_MIN = 0.5
OBJECT_MASS_MAX = 2.0

# Collision Settings
COLLISION_ENABLED = True
COEFFICIENT_RESTITUTION = 0.7  # Bounciness (0-1)
COEFFICIENT_FRICTION = 0.3
COLLISION_THRESHOLD = 0.001

# ============================================================================
# PARTICLE SYSTEM SETTINGS
# ============================================================================

PARTICLE_EMITTER_COUNT = 2
PARTICLES_PER_EMITTER = 100
PARTICLE_LIFETIME = 2.0  # seconds
PARTICLE_SPEED = 5.0
PARTICLE_SIZE = 0.1

# ============================================================================
# RENDERING SETTINGS
# ============================================================================

# Camera
CAMERA_SPEED = 15.0  # units/second
CAMERA_SENSITIVITY = 0.005  # radians
CAMERA_FOV = 45.0  # degrees
CAMERA_NEAR = 0.1
CAMERA_FAR = 1000.0

# Lighting
AMBIENT_STRENGTH = 0.3
DIFFUSE_STRENGTH = 0.7
SPECULAR_STRENGTH = 0.5
SHININESS = 32.0

# Shadow Mapping
SHADOW_MAP_WIDTH = 2048
SHADOW_MAP_HEIGHT = 2048
SHADOW_BIAS = 0.005

# Post-Processing
ENABLE_SHADOWS = True
ENABLE_BLOOM = False  # GPU intensive
ENABLE_NORMAL_MAPPING = True

# ============================================================================
# TIME SYSTEM SETTINGS
# ============================================================================

MAX_HISTORY_FRAMES = 3600  # 60 seconds at 60 FPS
TIME_SCALE_MIN = 0.1
TIME_SCALE_MAX = 2.0
TIME_SCALE_DEFAULT = 1.0

# ============================================================================
# UI SETTINGS
# ============================================================================

UI_FONT_SIZE = 32
UI_SMALL_FONT_SIZE = 20
UI_PADDING = 10
SHOW_DEBUG_INFO = True
SHOW_FPS_COUNTER = True

# ============================================================================
# COLOR PALETTE
# ============================================================================

class Colors:
    """Color definitions for UI and rendering"""
    # Background
    BG_DARK = (0.05, 0.05, 0.1)
    BG_DARKER = (0.02, 0.02, 0.05)
    
    # UI
    PRIMARY = (0.4, 0.8, 1.0)      # Light blue
    ACCENT = (1.0, 0.4, 0.8)       # Pink
    SUCCESS = (0.4, 1.0, 0.6)      # Green
    WARNING = (1.0, 0.8, 0.4)      # Orange
    DANGER = (1.0, 0.4, 0.4)       # Red
    
    # Text
    TEXT_PRIMARY = (0.9, 0.9, 1.0)
    TEXT_SECONDARY = (0.6, 0.7, 0.9)
    
    # Objects
    OBJECT_CUBE = (0.8, 0.3, 0.3)
    OBJECT_SPHERE = (0.3, 0.8, 0.3)
    OBJECT_SELECTED = (1.0, 1.0, 0.2)
    
    # Lighting
    LIGHT_COLOR = (1.0, 1.0, 1.0)
    LIGHT_AMBIENT = (0.3, 0.3, 0.4)

# ============================================================================
# SIMULATION MODES
# ============================================================================

class SimMode(Enum):
    """Simulation playback modes"""
    PLAYING = 1
    PAUSED = 2
    REVERSING = 3

class CameraMode(Enum):
    """Camera viewing modes"""
    FREE = 1
    ORBITAL = 2
    FIXED = 3
    CINEMATIC = 4

# ============================================================================
# DEBUG FLAGS
# ============================================================================

DEBUG_PHYSICS = False
DEBUG_RENDERING = False
DEBUG_COLLISIONS = False
DEBUG_PERFORMANCE = False

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# LOD (Level of Detail)
LOD_ENABLED = True
LOD_DISTANCE_NEAR = 10.0
LOD_DISTANCE_FAR = 100.0

# Culling
FRUSTUM_CULLING = True
BACKFACE_CULLING = True

# Memory
OBJECT_POOL_SIZE = 100
PARTICLE_POOL_SIZE = 10000

print("[ok] Configuration loaded successfully")

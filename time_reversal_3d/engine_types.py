"""
Data structures and types for Time-Reversal Physics Simulator
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

# ============================================================================
# TRANSFORM DATA
# ============================================================================

@dataclass
class Vector3:
    """3D Vector"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)
    
    @staticmethod
    def from_numpy(arr: np.ndarray) -> 'Vector3':
        return Vector3(float(arr[0]), float(arr[1]), float(arr[2]))


@dataclass
class Quaternion:
    """Rotation quaternion (x, y, z, w)"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0
    
    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.w], dtype=np.float32)
    
    @staticmethod
    def identity() -> 'Quaternion':
        return Quaternion(0, 0, 0, 1)


@dataclass
class Transform:
    """3D Transformation (position, rotation, scale)"""
    position: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion.identity)
    scale: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))
    
    def get_matrix(self) -> np.ndarray:
        """Compute transformation matrix"""
        # Simplified: would compute full model matrix with rotation
        pos_mat = np.eye(4, dtype=np.float32)
        pos_mat[:3, 3] = self.position.to_numpy()
        
        scale_mat = np.eye(4, dtype=np.float32)
        scale_mat[0, 0] = self.scale.x
        scale_mat[1, 1] = self.scale.y
        scale_mat[2, 2] = self.scale.z
        
        return pos_mat @ scale_mat


# ============================================================================
# PHYSICS DATA
# ============================================================================

@dataclass
class PhysicsState:
    """Complete physics state for a rigidbody at one moment in time"""
    position: np.ndarray  # [3] float32
    rotation: np.ndarray  # [4] float32 (quaternion)
    velocity: np.ndarray  # [3] float32
    angular_velocity: np.ndarray  # [3] float32
    timestamp: float = 0.0
    frame_index: int = 0
    
    def copy(self) -> 'PhysicsState':
        """Deep copy of state"""
        return PhysicsState(
            position=self.position.copy(),
            rotation=self.rotation.copy(),
            velocity=self.velocity.copy(),
            angular_velocity=self.angular_velocity.copy(),
            timestamp=self.timestamp,
            frame_index=self.frame_index
        )


@dataclass
class RigidbodyData:
    """Rigidbody physics component data"""
    mass: float = 1.0
    inv_mass: float = 1.0
    
    # Inertia tensor
    inertia: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    inv_inertia: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    
    # Linear motion
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    
    # Angular motion
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    angular_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    
    # Material properties
    restitution: float = 0.7  # Bounciness
    friction: float = 0.3


@dataclass
class CollisionData:
    """Information about a collision"""
    obj_a_id: int
    obj_b_id: int
    contact_point: np.ndarray  # [3] position of collision
    normal: np.ndarray  # [3] collision normal
    depth: float  # penetration depth
    impulse: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


# ============================================================================
# RENDERING DATA
# ============================================================================

@dataclass
class Mesh:
    """3D Mesh data"""
    vertices: np.ndarray  # [N, 3] float32
    indices: np.ndarray   # [M] uint32
    normals: np.ndarray   # [N, 3] float32
    
    VAO: int = 0  # Vertex Array Object
    VBO: int = 0  # Vertex Buffer Object
    EBO: int = 0  # Element Buffer Object
    vertex_count: int = 0


@dataclass
class Material:
    """Material properties for rendering"""
    ambient: Tuple[float, float, float]
    diffuse: Tuple[float, float, float]
    specular: Tuple[float, float, float]
    shininess: float = 32.0
    
    def to_array(self) -> np.ndarray:
        return np.array([
            *self.ambient,
            *self.diffuse,
            *self.specular,
            self.shininess
        ], dtype=np.float32)


# ============================================================================
# PARTICLE DATA
# ============================================================================

@dataclass
class Particle:
    """Individual particle properties"""
    position: np.ndarray  # [3] float32
    velocity: np.ndarray  # [3] float32
    acceleration: np.ndarray  # [3] float32
    lifetime: float  # seconds
    max_lifetime: float  # seconds
    size: float
    color: np.ndarray  # [4] RGBA
    
    def get_alpha(self) -> float:
        """Get alpha transparency based on lifetime"""
        return max(0.0, self.lifetime / self.max_lifetime)


# ============================================================================
# GAME OBJECT DATA
# ============================================================================

@dataclass
class GameObject:
    """Complete game object definition"""
    id: int
    name: str
    
    # Transform
    position: np.ndarray  # [3]
    rotation: np.ndarray  # [4] quaternion
    scale: np.ndarray  # [3]
    
    # Physics
    physics_enabled: bool = True
    mass: float = 1.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    
    # Rendering
    mesh: Optional[Mesh] = None
    material: Optional[Material] = None
    
    # Physics state history
    history: List[PhysicsState] = field(default_factory=list)
    
    def get_state(self, frame_idx: int = -1) -> PhysicsState:
        """Get physics state at given frame"""
        if not self.history or frame_idx >= len(self.history):
            frame_idx = -1
        
        state = self.history[frame_idx]
        return state


# ============================================================================
# CAMERA DATA
# ============================================================================

@dataclass
class CameraData:
    """Camera state"""
    position: np.ndarray  # [3]
    forward: np.ndarray   # [3]
    right: np.ndarray     # [3]
    up: np.ndarray        # [3]
    
    fov: float = 45.0
    aspect_ratio: float = 16.0 / 9.0
    near: float = 0.1
    far: float = 1000.0
    
    def get_view_matrix(self) -> np.ndarray:
        """Compute view matrix"""
        center = self.position + self.forward
        
        # Build view matrix manually
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = self.right
        view[1, :3] = self.up
        view[2, :3] = -self.forward
        
        view[0, 3] = -np.dot(self.right, self.position)
        view[1, 3] = -np.dot(self.up, self.position)
        view[2, 3] = np.dot(self.forward, self.position)
        
        return view
    
    def get_projection_matrix(self) -> np.ndarray:
        """Compute projection matrix (perspective)"""
        f = 1.0 / np.tan(np.radians(self.fov) / 2.0)
        range_inv = 1.0 / (self.near - self.far)
        
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / self.aspect_ratio
        proj[1, 1] = f
        proj[2, 2] = (self.near + self.far) * range_inv
        proj[2, 3] = -1.0
        proj[3, 2] = 2.0 * self.near * self.far * range_inv
        
        return proj

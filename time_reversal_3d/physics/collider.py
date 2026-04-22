"""
Collision detection and response
"""

import numpy as np
from enum import Enum
from typing import List, Tuple, Optional
from dataclasses import dataclass
from ..engine_types import CollisionData

class ColliderType(Enum):
    """Types of colliders"""
    SPHERE = 1
    BOX = 2
    PLANE = 3

@dataclass
class Collider:
    """Base collider class"""
    collider_type: ColliderType
    position: np.ndarray  # [3]
    is_trigger: bool = False
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get AABB bounds (min, max)"""
        raise NotImplementedError


@dataclass
class SphereCollider(Collider):
    """Sphere collider"""
    radius: float = 1.0
    
    def __init__(self, position: np.ndarray, radius: float = 1.0):
        super().__init__(ColliderType.SPHERE, position.copy())
        self.radius = radius
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get AABB bounds"""
        min_bound = self.position - self.radius
        max_bound = self.position + self.radius
        return min_bound, max_bound
    
    def closest_point(self, point: np.ndarray) -> np.ndarray:
        """Get closest point on sphere to external point"""
        delta = point - self.position
        distance = np.linalg.norm(delta)
        if distance <= self.radius:
            return point
        return self.position + (delta / distance) * self.radius


@dataclass
class BoxCollider(Collider):
    """Axis-aligned box collider"""
    half_extents: np.ndarray  # [3]
    
    def __init__(self, position: np.ndarray, half_extents: np.ndarray):
        super().__init__(ColliderType.BOX, position.copy())
        self.half_extents = half_extents.copy()
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get AABB bounds"""
        return self.position - self.half_extents, self.position + self.half_extents
    
    def closest_point(self, point: np.ndarray) -> np.ndarray:
        """Get closest point on box to external point"""
        return np.clip(point, 
                      self.position - self.half_extents,
                      self.position + self.half_extents)


@dataclass
class PlaneCollider(Collider):
    """Infinite plane collider"""
    normal: np.ndarray = np.array([0, 1, 0], dtype=np.float32)  # [3]
    
    def __init__(self, position: np.ndarray, normal: Optional[np.ndarray] = None):
        super().__init__(ColliderType.PLANE, position.copy())
        if normal is not None:
            self.normal = normal.copy()
            self.normal /= np.linalg.norm(self.normal)
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Infinite bounds"""
        return np.array([-1e9, -1e9, -1e9]), np.array([1e9, 1e9, 1e9])


class CollisionDetector:
    """Handles collision detection and response"""
    
    @staticmethod
    def sphere_sphere(c1: SphereCollider, c2: SphereCollider) -> Optional[CollisionData]:
        """Check sphere-sphere collision"""
        delta = c2.position - c1.position
        distance = np.linalg.norm(delta)
        min_distance = c1.radius + c2.radius
        
        if distance >= min_distance:
            return None  # No collision
        
        if distance < 0.0001:
            delta = np.array([1, 0, 0], dtype=np.float32)
            distance = 1.0
        
        normal = delta / distance
        penetration = min_distance - distance
        contact_point = c1.position + normal * c1.radius
        
        return CollisionData(
            obj_a_id=0,  # Would be set by collision system
            obj_b_id=0,
            contact_point=contact_point,
            normal=normal,
            depth=penetration
        )
    
    @staticmethod
    def sphere_plane(sphere: SphereCollider, plane: PlaneCollider) -> Optional[CollisionData]:
        """Check sphere-plane collision"""
        # Distance from sphere center to plane
        delta = sphere.position - plane.position
        distance_to_plane = np.dot(delta, plane.normal)
        
        if distance_to_plane >= sphere.radius:
            return None  # No collision
        
        penetration = sphere.radius - distance_to_plane
        contact_point = sphere.position - plane.normal * distance_to_plane
        
        return CollisionData(
            obj_a_id=0,
            obj_b_id=0,
            contact_point=contact_point,
            normal=plane.normal.copy(),
            depth=penetration
        )
    
    @staticmethod
    def box_box(c1: BoxCollider, c2: BoxCollider) -> Optional[CollisionData]:
        """Check box-box collision (AABB)"""
        min1, max1 = c1.get_bounds()
        min2, max2 = c2.get_bounds()
        
        # Check all axes
        for axis in range(3):
            if max1[axis] < min2[axis] or max2[axis] < min1[axis]:
                return None  # Separated on this axis
        
        # Collision detected
        min_overlap = float('inf')
        collision_normal = np.zeros(3, dtype=np.float32)
        
        # Find collision normal and depth
        for axis in range(3):
            overlap1 = max1[axis] - min2[axis]
            overlap2 = max2[axis] - min1[axis]
            min_o = min(overlap1, overlap2)
            
            if min_o < min_overlap:
                min_overlap = min_o
                collision_normal = np.zeros(3, dtype=np.float32)
                if overlap1 < overlap2:
                    collision_normal[axis] = -1
                else:
                    collision_normal[axis] = 1
        
        # Contact point (simplified)
        contact_point = (c1.position + c2.position) / 2.0
        
        return CollisionData(
            obj_a_id=0,
            obj_b_id=0,
            contact_point=contact_point,
            normal=collision_normal,
            depth=min_overlap
        )
    
    @staticmethod
    def aabb_aabb(min1: np.ndarray, max1: np.ndarray,
                  min2: np.ndarray, max2: np.ndarray) -> bool:
        """Quick AABB test"""
        return not (max1[0] < min2[0] or max1[1] < min2[1] or max1[2] < min2[2] or
                    max2[0] < min1[0] or max2[1] < min1[1] or max2[2] < min1[2])

"""
Main physics simulation engine
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import math

from ..config import GRAVITY, PHYSICS_TIMESTEP, COLLISION_THRESHOLD
from ..engine_types import PhysicsState, GameObject, CollisionData
from .rigidbody import Rigidbody
from .collider import SphereCollider, PlaneCollider, BoxCollider, CollisionDetector

@dataclass
class CollisionPair:
    """Pair of objects in collision"""
    obj_a: GameObject
    obj_b: GameObject
    collision_data: CollisionData


class PhysicsEngine:
    """Complete physics simulation engine"""
    
    def __init__(self):
        self.gravity = np.array([0, -GRAVITY, 0], dtype=np.float32)
        self.timestep = PHYSICS_TIMESTEP
        self.objects: List[GameObject] = []
        self.collisions: List[CollisionPair] = []
        self.detector = CollisionDetector()
        
        # Create ground plane
        self.ground_plane = PlaneCollider(
            np.array([0, -5, 0], dtype=np.float32),
            np.array([0, 1, 0], dtype=np.float32)
        )
    
    def add_object(self, obj: GameObject):
        """Add object to physics simulation"""
        self.objects.append(obj)
    
    def remove_object(self, obj: GameObject):
        """Remove object from physics simulation"""
        if obj in self.objects:
            self.objects.remove(obj)
    
    def update(self, delta_time: float = None):
        """Perform one physics timestep"""
        if delta_time is None:
            delta_time = self.timestep
        
        # Apply forces and update velocities
        self._apply_forces(delta_time)
        
        # Update positions
        self._integrate(delta_time)
        
        # Detect collisions
        self.collisions.clear()
        self._collision_detection()
        
        # Resolve collisions
        self._resolve_collisions()
        
        # Store frame state (for time reversal)
        self._store_frame_state()
    
    def _apply_forces(self, dt: float):
        """Apply all forces to objects"""
        for obj in self.objects:
            if not obj.physics_enabled:
                continue
            
            rb = Rigidbody(mass=obj2rigidbody(obj).mass)
            
            # Reset forces
            forces = np.zeros(3, dtype=np.float32)
            
            # Gravity
            if hasattr(obj, 'rigidbody') and obj.rigidbody.use_gravity:
                forces += self.gravity * obj.rigidbody.mass
            
            # Apply forces
            if hasattr(obj, 'rigidbody'):
                obj.rigidbody.forces = forces
                obj.rigidbody.update_acceleration()
    
    def _integrate(self, dt: float):
        """Integrate velocity and position"""
        for obj in self.objects:
            if not obj.physics_enabled:
                continue
            
            if not hasattr(obj, 'rigidbody'):
                continue
            
            rb = obj.rigidbody
            
            # Update velocity: v = v + a*dt
            rb.velocity += rb.acceleration * dt
            
            # Apply damping
            rb.velocity *= 0.99  # Simple damping
            
            # Update position: p = p + v*dt
            obj.position += rb.velocity * dt
    
    def _collision_detection(self):
        """Detect collisions between objects"""
        # Object-object collisions
        for i in range(len(self.objects)):
            for j in range(i + 1, len(self.objects)):
                obj_a = self.objects[i]
                obj_b = self.objects[j]
                
                if not obj_a.physics_enabled or not obj_b.physics_enabled:
                    continue
                
                collision = self._check_collision(obj_a, obj_b)
                if collision:
                    collision.obj_a_id = i
                    collision.obj_b_id = j
                    self.collisions.append(CollisionPair(obj_a, obj_b, collision))
        
        # Object-ground collisions
        for i, obj in enumerate(self.objects):
            if not obj.physics_enabled:
                continue
            
            # Create sphere collider from object
            size = np.linalg.norm(obj.scale) * 0.5
            collider = SphereCollider(obj.position, size)
            
            collision = self.detector.sphere_plane(collider, self.ground_plane)
            if collision and collision.depth > COLLISION_THRESHOLD:
                collision.obj_a_id = i
                collision.obj_b_id = -1  # Ground
                
                # Create dummy ground object
                ground_obj = GameObject(
                    id=-1, name="Ground",
                    position=np.array([0, -5, 0], dtype=np.float32),
                    rotation=np.array([0, 0, 0, 1], dtype=np.float32),
                    scale=np.array([1, 1, 1], dtype=np.float32)
                )
                ground_obj.physics_enabled = False
                
                self.collisions.append(CollisionPair(obj, ground_obj, collision))
    
    def _check_collision(self, obj_a: GameObject, obj_b: GameObject) -> Optional[CollisionData]:
        """Check collision between two objects"""
        size_a = np.linalg.norm(obj_a.scale) * 0.5
        size_b = np.linalg.norm(obj_b.scale) * 0.5
        
        # Create colliders
        sphere_a = SphereCollider(obj_a.position, size_a)
        sphere_b = SphereCollider(obj_b.position, size_b)
        
        return self.detector.sphere_sphere(sphere_a, sphere_b)
    
    def _resolve_collisions(self):
        """Resolve all detected collisions"""
        for pair in self.collisions:
            if pair.obj_a.physics_enabled and pair.obj_b.physics_enabled:
                self._resolve_collision_pair(pair)
    
    def _resolve_collision_pair(self, pair: CollisionPair):
        """Resolve collision between two objects"""
        obj_a = pair.obj_a
        obj_b = pair.obj_b
        collision = pair.collision_data
        
        if not hasattr(obj_a, 'rigidbody') or not hasattr(obj_b, 'rigidbody'):
            return
        
        rb_a = obj_a.rigidbody
        rb_b = obj_b.rigidbody
        
        # Separate penetrating objects
        if collision.depth > 0:
            separation = collision.normal * (collision.depth / 2.0 + 0.01)
            obj_a.position -= separation
            obj_b.position += separation
        
        # Calculate collision response
        rel_velocity = rb_a.velocity - rb_b.velocity
        velocity_along_normal = np.dot(rel_velocity, collision.normal)
        
        # Don't resolve if objects are separating
        if velocity_along_normal >= 0:
            return
        
        # Calculate impulse
        restitution = rb_a.restitution * rb_b.restitution
        impulse_magnitude = -(1 + restitution) * velocity_along_normal
        impulse_magnitude /= (rb_a.inverse_mass + rb_b.inverse_mass)
        
        impulse = collision.normal * impulse_magnitude
        
        # Apply impulse
        rb_a.velocity += impulse * rb_a.inverse_mass
        rb_b.velocity -= impulse * rb_b.inverse_mass
    
    def _store_frame_state(self):
        """Store physics state of all objects for time reversal"""
        for obj in self.objects:
            if not obj.physics_enabled:
                continue
            
            if not hasattr(obj, 'rigidbody'):
                continue
            
            state = PhysicsState(
                position=obj.position.copy(),
                rotation=obj.rotation.copy(),
                velocity=obj.rigidbody.velocity.copy(),
                angular_velocity=obj.rigidbody.angular_velocity.copy(),
                frame_index=len(obj.history)
            )
            
            obj.history.append(state)
            
            # Limit history to prevent memory overflow
            max_history = 3600  # 60 seconds at 60 FPS
            if len(obj.history) > max_history:
                obj.history.pop(0)
    
    def get_num_objects(self) -> int:
        """Get number of objects in simulation"""
        return len(self.objects)
    
    def get_collisions(self) -> List[CollisionPair]:
        """Get current collisions"""
        return self.collisions.copy()


def obj2rigidbody(obj: GameObject) -> Rigidbody:
    """Extract or create rigidbody from GameObject"""
    if hasattr(obj, 'rigidbody'):
        return obj.rigidbody
    
    rb = Rigidbody(mass=obj.mass if hasattr(obj, 'mass') else 1.0)
    obj.rigidbody = rb
    return rb

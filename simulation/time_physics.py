# -*- coding: utf-8 -*-
"""
time_physics.py — Time-Reversal Physics Engine
===============================================
Complete physics simulation with temporal mechanics for time-reversal capabilities
"""

from dataclasses import dataclass, field
from collections import deque
from typing import List, Optional, Tuple
import numpy as np
import math


# ═════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PhysicsState:
    """Complete snapshot of a physics object state at a point in time"""
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    rotation: float
    angular_velocity: float
    timestamp: float
    
    def copy(self):
        """Create a deep copy of this state"""
        return PhysicsState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy(),
            rotation=self.rotation,
            angular_velocity=self.angular_velocity,
            timestamp=self.timestamp
        )


class PhysicsObject:
    """Represents a single physics-simulated object"""
    _next_id = 1
    
    def __init__(self, position: np.ndarray, velocity: np.ndarray, 
                 acceleration: np.ndarray = None,
                 mass: float = 1.0, radius: float = 0.5, 
                 color: np.ndarray = None, name: str = "Object"):
        """
        Initialize a physics object
        
        Args:
            position: 3D position [x, y, z]
            velocity: 3D velocity [vx, vy, vz]
            mass: Object mass (affects acceleration)
            radius: Collision radius
            color: RGB color [0-1]
            name: Object identifier
        """
        self.id = PhysicsObject._next_id
        PhysicsObject._next_id += 1
        self.mass = max(0.1, mass)  # Prevent zero mass
        self.radius = max(0.1, radius)
        self.name = name
        self.fixed = False  # If True, object doesn't move
        
        # Physics state
        self.position = position.astype(np.float32)
        self.velocity = velocity.astype(np.float32)
        self.acceleration = acceleration.astype(np.float32) if acceleration is not None else np.zeros(3, dtype=np.float32)
        self.rotation = 0.0
        self.angular_velocity = 0.0
        
        # Rendering
        self.color = color if color is not None else np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.age = 0.0
        
        # Physics properties
        self.drag = 0.01  # Air resistance
        self.restitution = 0.85  # Bounciness (0-1)
        self.inertia = (2.0 / 5.0) * self.mass * (self.radius ** 2)
        
        # History
        self.state_history: deque = deque(maxlen=300)  # Store last 300 states
    
    def apply_force(self, force: np.ndarray):
        """Apply a force to the object (F = ma, so a = F/m)"""
        if not self.fixed:
            self.acceleration += force / self.mass
    
    def apply_gravity(self, gravity: float = -9.81):
        """Apply gravity (downward acceleration)"""
        if not self.fixed:
            self.apply_force(np.array([0, gravity * self.mass, 0], dtype=np.float32))
    
    def save_state(self, timestamp: float):
        """Save current state for time-reversal"""
        state = PhysicsState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy(),
            rotation=self.rotation,
            angular_velocity=self.angular_velocity,
            timestamp=timestamp
        )
        self.state_history.append(state)
    
    def restore_state(self, state: PhysicsState):
        """Restore object to a previous state"""
        self.position = state.position.copy()
        self.velocity = state.velocity.copy()
        self.acceleration = state.acceleration.copy()
        self.rotation = state.rotation
        self.angular_velocity = state.angular_velocity
    
    def interpolate_state(self, state1: PhysicsState, state2: PhysicsState, t: float):
        """Interpolate between two states (for smooth playback)"""
        # t ranges from 0 (at state1) to 1 (at state2)
        inv_t = 1.0 - t
        self.position = inv_t * state1.position + t * state2.position
        self.velocity = inv_t * state1.velocity + t * state2.velocity
        self.rotation = inv_t * state1.rotation + t * state2.rotation
    
    def update(self, dt: float):
        """Update object physics using Verlet integration"""
        if self.fixed:
            return
        
        # Apply velocity damping
        self.velocity *= (1.0 - self.drag)
        
        # Verlet integration: x = x + v*dt + 0.5*a*dt²
        self.position += self.velocity * dt + 0.5 * self.acceleration * dt * dt
        self.velocity += self.acceleration * dt
        
        # Reset acceleration each frame
        self.acceleration.fill(0)
        
        # Update rotation
        self.rotation += self.angular_velocity * dt
        self.age += dt


# ═════════════════════════════════════════════════════════════════════════════
# PHYSICS ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TimeReversalEngine:
    """Main physics simulation engine with time-reversal capabilities"""
    
    def __init__(self, world_bounds: float = 20.0, gravity: float = -9.81, 
                 history_size: int = 600, max_objects: int = 100):
        """
        Initialize the physics engine
        
        Args:
            world_bounds: Size of simulation space (cubic bounds)
            gravity: Gravitational acceleration
            history_size: How many frames to store for time-reversal
            max_objects: Maximum number of physics objects
        """
        self.world_bounds = world_bounds
        self.gravity = gravity
        self.history_size = history_size
        self.max_objects = max_objects
        
        # Objects and time
        self.objects: List[PhysicsObject] = []
        self.frame_history: deque = deque(maxlen=history_size)
        self.current_frame = 0
        self.total_time = 0.0
        self.dt = 0.016  # 60 FPS
        
        # Time control
        self.time_scale = 1.0  # Normal playback speed
        self.recording = True
        self.is_reversing = False
        self.paused = False
        
        # Statistics
        self.total_collisions = 0
        self.total_energy = 0.0
        self.total_kinetic_energy = 0.0
        self.total_potential_energy = 0.0
        self.collision_pairs = []
    
    def add_object(self, obj: PhysicsObject):
        """Add object to simulation"""
        if len(self.objects) < self.max_objects:
            if not hasattr(obj, "id"):
                obj.id = PhysicsObject._next_id
                PhysicsObject._next_id += 1
            self.objects.append(obj)
            return True
        return False
    
    def remove_object(self, index: int):
        """Remove object by index"""
        if 0 <= index < len(self.objects):
            self.objects.pop(index)
    
    def clear_objects(self):
        """Clear all objects"""
        self.objects.clear()
        self.frame_history.clear()
        self.current_frame = 0

    @property
    def history(self):
        """Compatibility alias for advanced UI code."""
        return self.frame_history

    @property
    def max_history(self):
        """Compatibility alias for advanced UI code."""
        return self.history_size

    def clear_history(self):
        """Clear timeline history while preserving objects."""
        self.frame_history.clear()
        for obj in self.objects:
            obj.state_history.clear()
        self.current_frame = 0

    def rewind_timeline(self, steps: int = 1):
        """Move backward through recorded timeline states."""
        if not self.frame_history:
            return False
        return self.reverse_to_frame(max(0, len(self.frame_history) - 1 - int(steps)))

    def fastforward_timeline(self, steps: int = 1):
        """Move forward through recorded timeline states."""
        if not self.frame_history:
            return False
        return self.reverse_to_frame(min(len(self.frame_history) - 1, int(self.current_frame) + int(steps)))

    def scrub_timeline(self, normalized_delta: float):
        """Scrub through history by a normalized screen delta."""
        if not self.frame_history:
            return False
        target = int(np.clip(self.current_frame + normalized_delta * len(self.frame_history), 0, len(self.frame_history) - 1))
        return self.reverse_to_frame(target)
    
    def apply_gravity(self):
        """Apply gravity to all non-fixed objects"""
        for obj in self.objects:
            obj.apply_gravity(self.gravity)
    
    def check_collisions(self) -> int:
        """
        Check for collisions between objects and boundaries
        Returns number of collisions detected
        """
        collisions = 0
        self.collision_pairs = []
        
        # Object-to-object collisions
        for i in range(len(self.objects)):
            for j in range(i + 1, len(self.objects)):
                if self._check_collision(self.objects[i], self.objects[j]):
                    self.collision_pairs.append((self.objects[i].id, self.objects[j].id))
                    collisions += 1
        
        # Boundary collisions
        for obj in self.objects:
            boundary_hits = self._check_boundary_collision(obj)
            if boundary_hits:
                self.collision_pairs.append((obj.id, "boundary"))
            collisions += boundary_hits
        
        self.total_collisions += collisions
        return collisions
    
    def _check_collision(self, obj1: PhysicsObject, obj2: PhysicsObject) -> bool:
        """Check if two objects collide"""
        min_dist = obj1.radius + obj2.radius
        dist = np.linalg.norm(obj2.position - obj1.position)
        
        if dist < min_dist:
            self._resolve_collision(obj1, obj2, dist, min_dist)
            return True
        return False
    
    def _resolve_collision(self, obj1: PhysicsObject, obj2: PhysicsObject, 
                          dist: float, min_dist: float):
        """Resolve collision between two objects using impulse-based method"""
        if dist < 0.001:
            dist = 0.001
        
        # Normal vector
        normal = (obj2.position - obj1.position) / dist
        
        # Relative velocity
        rel_vel = obj2.velocity - obj1.velocity
        vel_along_normal = np.dot(rel_vel, normal)
        
        # Don't collide if moving apart
        if vel_along_normal >= 0:
            return
        
        # Restitution (bounciness)
        restitution = min(obj1.restitution, obj2.restitution)
        
        # Impulse calculation
        inv_mass_sum = (1.0 / obj1.mass) + (1.0 / obj2.mass)
        impulse = -(1.0 + restitution) * vel_along_normal / inv_mass_sum
        
        # Apply impulse
        impulse_vec = impulse * normal
        if not obj1.fixed:
            obj1.velocity -= impulse_vec / obj1.mass
        if not obj2.fixed:
            obj2.velocity += impulse_vec / obj2.mass
        
        # Separate objects to prevent overlap
        separation = (min_dist - dist) / 2.0 + 0.001
        if not obj1.fixed and not obj2.fixed:
            obj1.position -= normal * separation
            obj2.position += normal * separation
        elif not obj1.fixed:
            obj1.position -= normal * (min_dist - dist + 0.001)
        elif not obj2.fixed:
            obj2.position += normal * (min_dist - dist + 0.001)
    
    def _check_boundary_collision(self, obj: PhysicsObject) -> int:
        """Check collisions with world boundaries"""
        collisions = 0
        b = self.world_bounds - obj.radius
        
        # X boundaries
        if obj.position[0] < -b:
            obj.position[0] = -b
            obj.velocity[0] *= -obj.restitution
            collisions += 1
        elif obj.position[0] > b:
            obj.position[0] = b
            obj.velocity[0] *= -obj.restitution
            collisions += 1
        
        # Y boundaries
        if obj.position[1] < -b:
            obj.position[1] = -b
            obj.velocity[1] *= -obj.restitution
            collisions += 1
        elif obj.position[1] > b:
            obj.position[1] = b
            obj.velocity[1] *= -obj.restitution
            collisions += 1
        
        # Z boundaries
        if obj.position[2] < -b:
            obj.position[2] = -b
            obj.velocity[2] *= -obj.restitution
            collisions += 1
        elif obj.position[2] > b:
            obj.position[2] = b
            obj.velocity[2] *= -obj.restitution
            collisions += 1
        
        return collisions
    
    def calculate_energy(self) -> float:
        """Calculate total kinetic + potential energy"""
        energy = 0.0
        kinetic_total = 0.0
        potential_total = 0.0
        for obj in self.objects:
            # Kinetic energy: 0.5 * m * v²
            kinetic = 0.5 * obj.mass * np.dot(obj.velocity, obj.velocity)
            # Potential energy: m * g * h (height)
            potential = obj.mass * abs(self.gravity) * obj.position[1]
            kinetic_total += kinetic
            potential_total += potential
            energy += kinetic + potential
        self.total_kinetic_energy = kinetic_total
        self.total_potential_energy = potential_total
        self.total_energy = energy
        return energy
    
    def update(self, dt: float = None, record: bool = True, collision_system=None):
        """
        Update physics simulation for one frame
        
        Args:
            dt: Delta time in seconds (uses default if not specified)
            record: Whether to record state for playback
        """
        if self.paused:
            return
        
        if dt is None:
            dt = self.dt

        if dt < 0:
            steps = max(1, int(abs(dt) / max(self.dt, 1e-6)))
            self.rewind_timeline(steps)
            return
        
        # Apply time scale
        actual_dt = dt * self.time_scale
        
        # Reset forces and apply gravity
        self.apply_gravity()
        
        # Update all objects
        for obj in self.objects:
            obj.update(actual_dt)
        
        # Check collisions
        self.check_collisions()
        self.calculate_energy()
        
        # Save state for time-reversal
        if record and self.recording:
            self._save_frame_state()
        
        # Update time counters
        self.total_time += actual_dt
        self.current_frame += 1
    
    def _save_frame_state(self):
        """Save current state of all objects for time-reversal"""
        states = []
        for obj in self.objects:
            obj.save_state(self.total_time)
            if obj.state_history:
                states.append(obj.state_history[-1])
        
        if states:
            self.frame_history.append({
                'timestamp': self.total_time,
                'frame': self.current_frame,
                'states': states
            })
    
    def reverse_to_frame(self, frame_num: int):
        """Go back to a specific frame in history"""
        if frame_num < 0 or frame_num >= len(self.frame_history):
            return False
        
        # Find the closest frame
        target_frame = list(self.frame_history)[frame_num]
        
        # Restore all objects
        for i, state in enumerate(target_frame['states']):
            if i < len(self.objects):
                self.objects[i].restore_state(state)
        
        self.total_time = target_frame['timestamp']
        self.current_frame = frame_num
        self.calculate_energy()
        return True
    
    def get_render_objects(self):
        """Get list of objects ready for rendering"""
        return [
            {
                'position': obj.position.copy(),
                'velocity': obj.velocity.copy(),
                'color': obj.color,
                'radius': obj.radius,
                'mass': obj.mass,
                'name': obj.name,
                'age': obj.age,
                'id': obj.id,
                'speed': float(np.linalg.norm(obj.velocity))
            }
            for obj in self.objects
        ]
    
    def get_statistics(self) -> dict:
        """Get simulation statistics"""
        total_mass = sum(obj.mass for obj in self.objects)
        return {
            'frame': self.current_frame,
            'time': self.total_time,
            'object_count': len(self.objects),
            'total_mass': total_mass,
            'total_collisions': self.total_collisions,
            'total_energy': self.calculate_energy(),
            'history_size': len(self.frame_history),
            'time_scale': self.time_scale,
            'recording': self.recording
        }

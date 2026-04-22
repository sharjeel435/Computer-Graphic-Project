"""
Rigidbody component for physics simulation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from ..engine_types import RigidbodyData, Vector3
from ..config import GRAVITY, COEFFICIENT_RESTITUTION, COEFFICIENT_FRICTION

@dataclass
class Rigidbody:
    """Physics rigidbody component"""
    
    mass: float = 1.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    
    # Material properties
    restitution: float = COEFFICIENT_RESTITUTION
    friction: float = COEFFICIENT_FRICTION
    use_gravity: bool = True
    is_kinematic: bool = False
    
    # Forces
    forces: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    torques: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    
    # Derived properties
    @property
    def inverse_mass(self) -> float:
        """Inverse mass (for infinite mass, set to 0)"""
        return 0.0 if self.is_kinematic else 1.0 / self.mass
    
    @property
    def kinetic_energy(self) -> float:
        """Total kinetic energy"""
        linear_ke = 0.5 * self.mass * np.dot(self.velocity, self.velocity)
        # Simplified (ignoring rotational)
        return linear_ke
    
    def apply_force(self, force: np.ndarray):
        """Apply force to rigidbody"""
        self.forces += force
    
    def apply_impulse(self, impulse: np.ndarray):
        """Apply instant velocity change"""
        if not self.is_kinematic and self.mass > 0:
            self.velocity += impulse / self.mass
    
    def apply_torque(self, torque: np.ndarray):
        """Apply rotational force"""
        self.torques += torque
    
    def reset_forces(self):
        """Reset accumulated forces (called each physics frame)"""
        self.forces = np.zeros(3, dtype=np.float32)
        self.torques = np.zeros(3, dtype=np.float32)
    
    def update_acceleration(self):
        """Calculate acceleration from forces"""
        if self.is_kinematic or self.mass <= 0:
            return
        
        # F = ma => a = F/m
        self.acceleration = self.forces / self.mass
        
        # Apply gravity
        if self.use_gravity:
            self.acceleration[1] -= GRAVITY
    
    def to_data(self) -> RigidbodyData:
        """Convert to RigidbodyData"""
        data = RigidbodyData()
        data.mass = self.mass
        data.inv_mass = self.inverse_mass
        data.velocity = self.velocity.copy()
        data.acceleration = self.acceleration.copy()
        data.angular_velocity = self.angular_velocity.copy()
        data.restitution = self.restitution
        data.friction = self.friction
        return data

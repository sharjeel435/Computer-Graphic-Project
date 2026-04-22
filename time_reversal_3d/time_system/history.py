"""
Time reversal and frame history management system
"""

import numpy as np
from typing import List, Optional, Dict
from ..engine_types import PhysicsState, GameObject
from ..config import MAX_HISTORY_FRAMES

class FrameHistory:
    """Efficient frame history storage using numpy arrays"""
    
    def __init__(self, max_frames: int = MAX_HISTORY_FRAMES):
        self.max_frames = max_frames
        self.current_frame_idx = 0
        self.frame_count = 0
        
        # Pre-allocate but use differently
        self.history: List[Dict[int, PhysicsState]] = []
    
    def store_frame(self, objects: List[GameObject]):
        """Store current frame state of all objects"""
        frame_data = {}
        
        for obj in objects:
            if not obj.physics_enabled or not hasattr(obj, 'rigidbody'):
                continue
            
            rb = obj.rigidbody
            state = PhysicsState(
                position=obj.position.copy(),
                rotation=obj.rotation.copy(),
                velocity=rb.velocity.copy(),
                angular_velocity=rb.angular_velocity.copy(),
                frame_index=self.frame_count
            )
            frame_data[obj.id] = state
        
        self.history.append(frame_data)
        self.current_frame_idx = len(self.history) - 1
        self.frame_count += 1
        
        # Limit history size
        if len(self.history) > self.max_frames:
            self.history.pop(0)
            self.current_frame_idx = max(0, self.current_frame_idx - 1)
    
    def get_frame(self, frame_idx: int, obj_id: int) -> Optional[PhysicsState]:
        """Retrieve state of object at given frame"""
        if frame_idx < 0 or frame_idx >= len(self.history):
            return None
        
        return self.history[frame_idx].get(obj_id)
    
    def get_all_states_at_frame(self, frame_idx: int) -> Dict[int, PhysicsState]:
        """Get all object states at frame"""
        if frame_idx < 0 or frame_idx >= len(self.history):
            return {}
        return self.history[frame_idx]
    
    def interpolate_state(self, state_a: PhysicsState, state_b: PhysicsState, 
                         t: float) -> PhysicsState:
        """Interpolate between two states (t in [0, 1])"""
        t = np.clip(t, 0, 1)
        
        # Linear interpolation for position and velocity
        pos = (1 - t) * state_a.position + t * state_b.position
        vel = (1 - t) * state_a.velocity + t * state_b.velocity
        
        # SLERP for rotation (simplified linear)
        rot = (1 - t) * state_a.rotation + t * state_b.rotation
        rot /= np.linalg.norm(rot)
        
        # Angular velocity
        ang_vel = (1 - t) * state_a.angular_velocity + t * state_b.angular_velocity
        
        return PhysicsState(
            position=pos,
            rotation=rot,
            velocity=vel,
            angular_velocity=ang_vel,
            frame_index=state_a.frame_index
        )
    
    def clear(self):
        """Clear history"""
        self.history.clear()
        self.current_frame_idx = 0
        self.frame_count = 0
    
    def get_frame_count(self) -> int:
        """Get total recorded frames"""
        return len(self.history)


class TimeReversal:
    """Manages time reversal playback"""
    
    def __init__(self, history: FrameHistory):
        self.history = history
        self.playback_frame = 0
        self.is_reversing = False
        self.playback_speed = 1.0
        self.accumulator = 0.0
    
    def start_reversal(self):
        """Begin time reversal from current frame"""
        self.is_reversing = True
        self.playback_frame = self.history.current_frame_idx
    
    def stop_reversal(self):
        """Stop time reversal"""
        self.is_reversing = False
    
    def update(self, delta_time: float):
        """Update playback frame"""
        if not self.is_reversing:
            return
        
        self.accumulator += delta_time * self.playback_speed
        
        # Reverse one frame per frame time
        if self.accumulator >= 1.0 / 60.0:  # 60 FPS playback
            self.playback_frame -= 1
            self.accumulator -= 1.0 / 60.0
            
            if self.playback_frame < 0:
                self.playback_frame = 0
                self.is_reversing = False
    
    def apply_reversed_state(self, objects: List[GameObject]):
        """Apply reversed frame state to objects"""
        if self.playback_frame >= len(self.history.history):
            return
        
        frame_data = self.history.get_all_states_at_frame(self.playback_frame)
        
        for obj in objects:
            if obj.id not in frame_data:
                continue
            
            state = frame_data[obj.id]
            obj.position = state.position.copy()
            obj.rotation = state.rotation.copy()
            if hasattr(obj, 'rigidbody'):
                obj.rigidbody.velocity = state.velocity.copy()
                obj.rigidbody.angular_velocity = state.angular_velocity.copy()


class MotionTrail:
    """Stores particle trail for motion visualization"""
    
    def __init__(self, max_length: int = 50):
        self.positions: List[np.ndarray] = []
        self.max_length = max_length
    
    def add_position(self, pos: np.ndarray):
        """Add position to trail"""
        self.positions.append(pos.copy())
        if len(self.positions) > self.max_length:
            self.positions.pop(0)
    
    def clear(self):
        """Clear trail"""
        self.positions.clear()
    
    def get_trail(self) -> List[np.ndarray]:
        """Get current trail"""
        return self.positions.copy()

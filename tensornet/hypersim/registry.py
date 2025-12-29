"""
Environment Registry
====================

Factory functions for creating and registering environments.

Usage:
    from tensornet.hypersim import make_env, register_env, list_envs
    
    env = make_env('fluid-control-v0')
    register_env('my-env-v0', MyEnvClass, default_config)
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Type, Callable, List

# Make gymnasium optional
try:
    import gymnasium
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gymnasium = None

from .env import FluidEnv, FluidEnvConfig


# =============================================================================
# REGISTRY
# =============================================================================

_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_env(
    env_id: str,
    entry_point: Type,
    default_config: Optional[Dict[str, Any]] = None,
    max_episode_steps: Optional[int] = None,
    kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Register an environment in the HyperSim registry.
    
    Args:
        env_id: Unique environment ID (e.g., 'fluid-control-v0')
        entry_point: Environment class
        default_config: Default configuration dict
        max_episode_steps: Maximum steps per episode
        kwargs: Additional kwargs passed to constructor
        
    Example:
        register_env(
            'my-fluid-v0',
            FluidEnv,
            {'viscosity': 0.01, 'grid_bits': 6},
        )
    """
    if env_id in _REGISTRY:
        raise ValueError(f"Environment {env_id} already registered")
    
    _REGISTRY[env_id] = {
        'entry_point': entry_point,
        'default_config': default_config or {},
        'max_episode_steps': max_episode_steps,
        'kwargs': kwargs or {},
    }


def make_env(
    env_id: str,
    render_mode: Optional[str] = None,
    **kwargs,
):
    """
    Create an environment from the registry.
    
    Args:
        env_id: Registered environment ID
        render_mode: 'human', 'rgb_array', or None
        **kwargs: Override default configuration
        
    Returns:
        Configured environment instance
        
    Example:
        env = make_env('fluid-control-v0', viscosity=0.02)
    """
    if env_id not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(
            f"Environment {env_id} not found. "
            f"Available: {available}"
        )
    
    spec = _REGISTRY[env_id]
    entry_point = spec['entry_point']
    
    # Merge configs: default < registry kwargs < user kwargs
    config = {**spec['default_config'], **spec['kwargs'], **kwargs}
    
    if render_mode is not None:
        config['render_mode'] = render_mode
    
    # Handle FluidEnv specially
    if entry_point == FluidEnv:
        env_config = FluidEnvConfig(**{
            k: v for k, v in config.items()
            if k in FluidEnvConfig.__dataclass_fields__
        })
        env = FluidEnv(config=env_config)
    else:
        env = entry_point(**config)
    
    # Apply time limit if specified
    if spec['max_episode_steps'] is not None:
        from .wrappers import TimeLimit
        env = TimeLimit(env, max_steps=spec['max_episode_steps'])
    
    return env


def list_envs() -> List[str]:
    """List all registered environment IDs."""
    return list(_REGISTRY.keys())


def get_env_spec(env_id: str) -> Dict[str, Any]:
    """Get environment specification."""
    if env_id not in _REGISTRY:
        raise ValueError(f"Environment {env_id} not found")
    return _REGISTRY[env_id].copy()


def unregister_env(env_id: str):
    """Remove an environment from the registry."""
    if env_id in _REGISTRY:
        del _REGISTRY[env_id]


# =============================================================================
# PRESET ENVIRONMENTS
# =============================================================================

def _register_presets():
    """Register preset environments."""
    
    # Basic fluid control
    register_env(
        'fluid-control-v0',
        FluidEnv,
        default_config={
            'ndim': 2,
            'bits_per_dim': 6,
            'rank': 8,
            'viscosity': 0.01,
            'dt': 0.01,
            'max_steps': 200,
            'obs_resolution': (64, 64),
        },
        max_episode_steps=200,
    )
    
    # Easy version (high viscosity, slow flow)
    register_env(
        'fluid-control-easy-v0',
        FluidEnv,
        default_config={
            'ndim': 2,
            'bits_per_dim': 5,
            'rank': 6,
            'viscosity': 0.1,
            'dt': 0.02,
            'max_steps': 100,
            'obs_resolution': (32, 32),
        },
        max_episode_steps=100,
    )
    
    # Hard version (low viscosity, turbulent)
    register_env(
        'fluid-control-hard-v0',
        FluidEnv,
        default_config={
            'ndim': 2,
            'bits_per_dim': 7,
            'rank': 12,
            'viscosity': 0.001,
            'dt': 0.005,
            'max_steps': 500,
            'obs_resolution': (64, 64),
        },
        max_episode_steps=500,
    )
    
    # 3D fluid control
    register_env(
        'fluid-control-3d-v0',
        FluidEnv,
        default_config={
            'ndim': 3,
            'bits_per_dim': 5,
            'rank': 8,
            'viscosity': 0.01,
            'dt': 0.01,
            'max_steps': 200,
            'obs_resolution': (32, 32, 32),
        },
        max_episode_steps=200,
    )
    
    # Smoke plume (buoyancy-driven)
    register_env(
        'smoke-plume-v0',
        FluidEnv,
        default_config={
            'ndim': 2,
            'bits_per_dim': 6,
            'rank': 8,
            'viscosity': 0.005,
            'dt': 0.01,
            'max_steps': 300,
            'obs_resolution': (64, 64),
            'include_buoyancy': True,
        },
        max_episode_steps=300,
    )
    
    # Vortex control
    register_env(
        'vortex-control-v0',
        FluidEnv,
        default_config={
            'ndim': 2,
            'bits_per_dim': 6,
            'rank': 10,
            'viscosity': 0.002,
            'dt': 0.01,
            'max_steps': 400,
            'obs_resolution': (64, 64),
        },
        max_episode_steps=400,
    )
    
    # Channel flow (boundary-driven)
    register_env(
        'channel-flow-v0',
        FluidEnv,
        default_config={
            'ndim': 2,
            'bits_per_dim': 6,
            'rank': 8,
            'viscosity': 0.01,
            'dt': 0.01,
            'max_steps': 200,
            'obs_resolution': (64, 32),
            'boundary_type': 'channel',
        },
        max_episode_steps=200,
    )


# Register presets on import
_register_presets()


# =============================================================================
# ENVIRONMENT MAKERS
# =============================================================================

def make_fluid_env(
    difficulty: str = 'medium',
    ndim: int = 2,
    render_mode: Optional[str] = None,
    **kwargs,
) -> FluidEnv:
    """
    Convenience function to create fluid control environment.
    
    Args:
        difficulty: 'easy', 'medium', or 'hard'
        ndim: Number of spatial dimensions (2 or 3)
        render_mode: 'human', 'rgb_array', or None
        **kwargs: Additional configuration overrides
        
    Returns:
        Configured FluidEnv
    """
    if difficulty == 'easy':
        base = 'fluid-control-easy-v0'
    elif difficulty == 'hard':
        base = 'fluid-control-hard-v0'
    else:
        base = 'fluid-control-v0'
    
    if ndim == 3:
        base = 'fluid-control-3d-v0'
    
    return make_env(base, render_mode=render_mode, **kwargs)


def make_vectorized_env(
    env_id: str,
    num_envs: int = 4,
    **kwargs,
):
    """
    Create vectorized environment for parallel training.
    
    Args:
        env_id: Environment ID
        num_envs: Number of parallel environments
        **kwargs: Environment configuration
        
    Returns:
        Vectorized environment
    """
    if not HAS_GYMNASIUM:
        raise ImportError("gymnasium is required for vectorized environments")
    
    def make_fn():
        return make_env(env_id, **kwargs)
    
    return gymnasium.vector.SyncVectorEnv([make_fn for _ in range(num_envs)])


# =============================================================================
# GYMNASIUM REGISTRATION
# =============================================================================

def register_with_gymnasium():
    """
    Register all HyperSim environments with Gymnasium.
    
    Allows using gymnasium.make('HyperSim/fluid-control-v0')
    """
    if not HAS_GYMNASIUM:
        return
    
    for env_id, spec in _REGISTRY.items():
        gym_id = f"HyperSim/{env_id}"
        
        try:
            gymnasium.register(
                id=gym_id,
                entry_point=f"tensornet.hypersim:make_env",
                kwargs={'env_id': env_id},
                max_episode_steps=spec.get('max_episode_steps'),
            )
        except gymnasium.error.Error:
            # Already registered
            pass


# =============================================================================
# TASK SPECIFICATIONS
# =============================================================================

class TaskSpec:
    """
    Specification for a control task.
    
    Defines goal, reward, and success criteria independently
    of the environment dynamics.
    """
    
    def __init__(
        self,
        name: str,
        goal_description: str,
        reward_type: str = 'dense',
        success_fn: Optional[Callable] = None,
        reward_fn: Optional[Callable] = None,
        initial_state_fn: Optional[Callable] = None,
    ):
        self.name = name
        self.goal_description = goal_description
        self.reward_type = reward_type
        self.success_fn = success_fn or (lambda s: False)
        self.reward_fn = reward_fn or (lambda s, a, ns: 0.0)
        self.initial_state_fn = initial_state_fn
    
    def is_success(self, state: Any) -> bool:
        """Check if task is complete."""
        return self.success_fn(state)
    
    def get_reward(
        self,
        state: Any,
        action: Any,
        next_state: Any,
    ) -> float:
        """Compute task-specific reward."""
        return self.reward_fn(state, action, next_state)


# Preset task specifications
TASK_SPECS = {
    'target_field': TaskSpec(
        name='target_field',
        goal_description='Match the target velocity field',
        reward_type='dense',
    ),
    'dissipation': TaskSpec(
        name='dissipation',
        goal_description='Minimize total kinetic energy',
        reward_type='dense',
    ),
    'vortex_creation': TaskSpec(
        name='vortex_creation',
        goal_description='Create a stable vortex at the target location',
        reward_type='shaped',
    ),
    'mixing': TaskSpec(
        name='mixing',
        goal_description='Maximize mixing between two fluid regions',
        reward_type='dense',
    ),
}

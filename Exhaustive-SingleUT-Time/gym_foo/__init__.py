try:
    from gymnasium.envs.registration import register as _register_gymnasium
except ImportError:  # pragma: no cover - fallback for gym-only installs
    _register_gymnasium = None

try:
    from gym.envs.registration import register as _register_gym
except ImportError:  # pragma: no cover
    _register_gym = None

def _safe_register(register_fn):
    if register_fn is None:
        return
    try:
        register_fn(
            id='foo-v0',
            entry_point='gym_foo.envs:FooEnv',
        )
    except Exception:
        pass

_safe_register(_register_gymnasium)
_safe_register(_register_gym)

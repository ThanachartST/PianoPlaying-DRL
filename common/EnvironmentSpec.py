from dataclasses import dataclass

import dm_env
import numpy as np
from dm_env import specs

# FIXME: For RNN Task we need to fixed this Spec Class
@dataclass(frozen=True)
class EnvironmentSpec: # FIXME: Change the name of this class
    # static_obs: specs.Array
    # seq_obs: specs.Array
    observation: specs.Array
    action: specs.Array

    @staticmethod
    def make(env: dm_env.Environment) -> "EnvironmentSpec":
        '''
        NOTE: We should change the action spec into BoundedArray.
        The previous version using specs.Array, which the minimum and maximum 
        of all action were in range [-1,1], but I realize that some joint 
        was not act like that.
        NOTE: This issue effect only warmup steps.
        '''

        obs_spec = env.observation_spec()
        action_spec = env.action_spec()
        print( f'##### observation spec #####' )
        # Check environment observation spec, it must come with dict instance
        if isinstance(obs_spec, dict):
            # Loop each key and value in observation spec
            for key, val in obs_spec.items():
                print( f'Key: { key }' )
                print( f'    { val }')

                # NOTE: I try to find the variable that defined the 'goal' key, but it have not.
                if key == 'goal':
                    # NOTE: n_keys plus 1 for current state
                    # NOTE: _n_steps_lookahead plus 1 for sustain pedal state
                    # FIXME: I dont even know how goal state come from
                    # And what sholud I reshape it. We need the expert to verify them
                    seq_obs_spec = specs.Array( shape = (env.task.piano.n_keys + 1, env.task._n_steps_lookahead + 1),
                                                dtype = np.float32,
                                                name = 'sequential_state' )
                else:
                    # Concat observation space

                    # Contribute it to specs.BoundedArray(shape=(???,), dtype=np.float32, name='static_state', minimum=-inf, maximum=inf)
                    pass

        else:
            print( obs_spec )
        print( f'##### action spec #####')
        print( action_spec )
        return EnvironmentSpec(
            observation=env.observation_spec(),
            action=env.action_spec(),
        )

    def sample_action(self, random_state: np.random.RandomState) -> np.ndarray:
        if not isinstance(self.action, specs.BoundedArray):
            raise ValueError("Only BoundedArray action specs are supported.")

        action = random_state.uniform(
            low=self.action.minimum, high=self.action.maximum, size=self.action.shape
        )
        return action.astype(self.action.dtype)

    @property
    def observation_dim(self) -> int:
        return self.observation.shape[-1]

    @property
    def action_dim(self) -> int:
        return self.action.shape[-1]


# def zeros_like(spec: specs.Array) -> jnp.ndarray:
#     return jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), spec)

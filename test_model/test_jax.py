# import jax
# import jax.numpy as jnp

# print("JAX devices:", jax.devices())
# x = jnp.ones((5000, 5000))
# print("x @ x.T =", jnp.dot(x, x.T))
import jax
import jaxlib
print("jax version:", jax.__version__)
print("jaxlib version:", jaxlib.__version__)
print("JAX devices:", jax.devices())

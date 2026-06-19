import jax
import jax.numpy as jnp
from flax import nnx

def sequential_forward(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    beta: jax.Array,
    gamma: jax.Array,
    delta: jax.Array
) -> jax.Array:
    batch_size, seq_len, query_dim = query.shape
    value_dim = value.shape[-1]

    S_t = jnp.zeros((batch_size, query_dim, value_dim), dtype=query.dtype)
    I = jnp.eye(query_dim, dtype=query.dtype)

    outputs: list[jax.Array] = []

    for t in range(seq_len):
        q_t = jnp.expand_dims(query[:, t, :], axis=1)   # (B, 1, d_k)
        k_t = jnp.expand_dims(key[:, t, :], axis=1)   # (B, 1, d_k)
        v_t = jnp.expand_dims(value[:, t, :], axis=1)   # (B, 1, d_v)
        b_t = jnp.expand_dims(beta[:, t, :], axis=1)   # (B, 1, d_k)
        w_t = jnp.expand_dims(gamma[:, t, :], axis=1)   # (B, 1, d_v)
        d_t = jnp.expand_dims(delta[:, t, :], axis=1)   # (B, 1, d_k)

        A_t = (I - k_t.swapaxes(1, 2) * (b_t * k_t)) * d_t
        B_t = k_t.swapaxes(1, 2) * (w_t * v_t)

        S_t = A_t @ S_t + B_t
        o_t = (q_t @ S_t).squeeze(1)

        outputs.append(o_t)

    return jnp.stack(outputs, axis=1)

def chunked_forward(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    beta: jax.Array,
    gamma: jax.Array,
    delta: jax.Array,
    chunk_size: int
) -> jax.Array:
    batch_size, seq_len, query_dim = query.shape
    value_dim = value.shape[-1]

    S_t = jnp.zeros((batch_size, query_dim, value_dim), dtype=query.dtype)
    I = jnp.broadcast_to(
        jnp.eye(query_dim, dtype=query.dtype),
        (batch_size, query_dim, query_dim)
    )

    outputs: list[jax.Array] = []
    num_chunks = seq_len // chunk_size

    for chunk_index in range(num_chunks):
        start = chunk_index * chunk_size

        q_c = jnp.expand_dims(query[:, start : start + chunk_size, :], axis=2)   # (B, 1, d_k)
        k_c = jnp.expand_dims(key[:, start : start + chunk_size, :], axis=2)   # (B, 1, d_k)
        v_c = jnp.expand_dims(value[:, start : start + chunk_size, :], axis=2)   # (B, 1, d_v)
        b_c = jnp.expand_dims(beta[:, start : start + chunk_size, :], axis=2)   # (B, 1, d_k)
        w_c = jnp.expand_dims(gamma[:, start : start + chunk_size, :], axis=2)   # (B, 1, d_v)
        d_c = jnp.expand_dims(delta[:, start : start + chunk_size, :], axis=2)   # (B, 1, d_k)

        A_c: list[jax.Array] = []
        B_c: list[jax.Array] = []

        for r in range(chunk_size):
            k_r = k_c[:, r, :, :] # (B, 1, d_k)
            v_r = v_c[:, r, :, :] # (B, 1, d_v)
            b_r = b_c[:, r, :, :] # (B, 1, d_k)
            w_r = w_c[:, r, :, :] # (B, 1, d_v)
            d_r = d_c[:, r, :, :] # (B, 1, d_k)

            A_r = (I - k_r.swapaxes(1, 2) * (b_r * k_r)) * d_r
            B_r = k_r.swapaxes(1, 2) * (w_r * v_r)

            A_c.append(A_r)
            B_c.append(B_r)

        o_c: list[jax.Array] = []
        S_c = S_t

        for r in range(chunk_size):
            prefix = I
            for i in range(r + 1):
                prefix = A_c[i] @ prefix

            sum = jnp.zeros((batch_size, query_dim, value_dim), dtype=query.dtype)
            for i in range(r + 1):
                suffix = I
                for j in range(i + 1, r + 1):
                    suffix = A_c[j] @ suffix

                sum += suffix @ B_c[i]

            S_r = prefix @ S_t + sum

            o_r = (q_c[:, r, :, :] @ S_r).squeeze(1)
            o_c.append(o_r)

            S_c = S_r

        S_t = S_c
        outputs.extend(o_c)

    return jnp.stack(outputs, axis=1)

def chunked_forward_optimized(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    beta: jax.Array,
    gamma: jax.Array,
    delta: jax.Array,
    chunk_size: int
) -> jax.Array:
    batch_size, seq_len, query_dim = query.shape
    value_dim = value.shape[-1]
    num_chunks = seq_len // chunk_size

    # 1. Reshape and transpose inputs to (num_chunks, batch, chunk_size, dim)
    # Transposing allows jax.lax.scan to iterate over the chunks (axis 0).
    def prepare_scan_input(x: jax.Array) -> jax.Array:
        x_reshaped = x.reshape(batch_size, num_chunks, chunk_size, -1)
        return x_reshaped.swapaxes(0, 1)

    q_scan = prepare_scan_input(query)
    k_scan = prepare_scan_input(key)
    v_scan = prepare_scan_input(value)
    b_scan = prepare_scan_input(beta)
    w_scan = prepare_scan_input(gamma)
    d_scan = prepare_scan_input(delta)

    I = jnp.eye(query_dim, dtype=query.dtype)

    # 2. Define the associative combination operator
    def combine(state1: tuple[jax.Array, jax.Array], state2: tuple[jax.Array, jax.Array]):
        A1, B1 = state1
        A2, B2 = state2

        # Batched matrix multiplications natively broadcast over the batch dimension
        A_out = A2 @ A1
        B_out = A2 @ B1 + B2
        return A_out, B_out

    # 3. Define the step function for the outer chunk scan
    def chunk_step(S_prev: jax.Array, xs: tuple):
        q_c, k_c, v_c, b_c, w_c, d_c = xs
        
        # Vectorized calculation of A_c and B_c for the entire chunk
        # outer_k shape: (batch, chunk_size, query_dim, query_dim)
        outer_k = jnp.einsum('bci,bcj->bcij', k_c, b_c * k_c)
        
        # d_c is expanded to broadcast across the columns of the matrix
        A_c = (I - outer_k) * jnp.expand_dims(d_c, axis=-2)
        B_c = jnp.einsum('bci,bcj->bcij', k_c, w_c * v_c)

        # Transpose chunk dimension to axis 0 for associative_scan
        A_c_t = A_c.swapaxes(0, 1)
        B_c_t = B_c.swapaxes(0, 1)

        # 4. Perform the associative scan to resolve the prefix matrices
        A_cum_t, B_cum_t = jax.lax.associative_scan(combine, (A_c_t, B_c_t))

        # Transpose back to (batch, chunk_size, ...)
        A_cum = A_cum_t.swapaxes(0, 1)
        B_cum = B_cum_t.swapaxes(0, 1)

        # 5. Compute the hidden states and outputs for all tokens in the chunk
        # S_all shape: (batch, chunk_size, query_dim, value_dim)
        S_all = jnp.einsum('bcij,bjk->bcik', A_cum, S_prev) + B_cum
        
        # o_c shape: (batch, chunk_size, value_dim)
        o_c = jnp.einsum('bci,bcij->bcj', q_c, S_all)

        # The state carried to the next chunk is the state of the final token
        S_next = S_all[:, -1, :]

        return S_next, o_c

    # Initial hidden state
    S_init = jnp.zeros((batch_size, query_dim, value_dim), dtype=query.dtype)

    # 6. Execute the compiled loop over the chunks
    _, o_scanned = jax.lax.scan(
        chunk_step,
        S_init,
        (q_scan, k_scan, v_scan, b_scan, w_scan, d_scan)
    )

    # Restore the original sequence dimension layout
    o_out = o_scanned.swapaxes(0, 1)
    return o_out.reshape(batch_size, num_chunks * chunk_size, value_dim)

# --- 1. The Gated DeltaNet Layer ---
class GatedDeltaNetLayer(nnx.Module):
    def __init__(self, dim: int, num_heads: int, chunk_size: int, rngs: nnx.Rngs):
        assert dim % num_heads == 0, "Dimension must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.chunk_size = chunk_size

        # Content Projections
        self.q_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)

        # Gate Projections (Erase, Write, Decay)
        self.beta_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.gamma_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.delta_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)

        # Output Projection
        self.o_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        batch_size, seq_len, dim = x.shape

        # 1. Project inputs
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. Project and activate gates
        # Beta (Erase) and Gamma (Write) are typically sigmoid or swish
        beta = jax.nn.sigmoid(self.beta_proj(x))
        gamma = jax.nn.sigmoid(self.gamma_proj(x))
        # Delta (Decay) MUST be strictly (0, 1) to ensure recurrent stability
        delta = jax.nn.sigmoid(self.delta_proj(x))

        # 3. Reshape for Multi-Head: (B, L, H, D_h)
        def reshape_to_heads(tensor):
            return tensor.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        q, k, v = map(reshape_to_heads, (q, k, v))
        
        # Unit-norm keys to prevent explosive values in I - outer_k
        k = k / jnp.maximum(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-6)

        beta, gamma, delta = map(reshape_to_heads, (beta, gamma, delta))

        # 4. Flatten Batch and Heads to reuse your exact chunked_forward function
        # Shape becomes: (B * H, L, D_h)
        def flatten_batch_heads(tensor):
            # Transpose to (B, H, L, D_h) then reshape to (B*H, L, D_h)
            tensor = jnp.swapaxes(tensor, 1, 2) 
            return tensor.reshape(batch_size * self.num_heads, seq_len, self.head_dim)

        q_flat, k_flat, v_flat = map(flatten_batch_heads, (q, k, v))
        b_flat, g_flat, d_flat = map(flatten_batch_heads, (beta, gamma, delta))

        # 5. Call your optimized recurrence function!
        # (Make sure chunked_forward_optimized is in scope)
        o_flat = chunked_forward_optimized(
            q_flat, k_flat, v_flat, 
            b_flat, g_flat, d_flat, 
            self.chunk_size
        )

        # 6. Unflatten back to Multi-Head: (B, H, L, D_h)
        o = o_flat.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        
        # 7. Recombine heads: (B, L, H, D_h) -> (B, L, Dim)
        o = jnp.swapaxes(o, 1, 2).reshape(batch_size, seq_len, dim)

        # 8. Final output projection
        return self.o_proj(o)

# --- 2. The Model Block (Attention + MLP) ---
class GatedDeltaNetBlock(nnx.Module):
    def __init__(self, dim: int, num_heads: int, mlp_dim: int, chunk_size: int, rngs: nnx.Rngs):
        # Normalization
        self.norm1 = nnx.RMSNorm(dim, rngs=rngs)
        self.norm2 = nnx.RMSNorm(dim, rngs=rngs)

        # Core DeltaNet Layer
        self.attn = GatedDeltaNetLayer(dim, num_heads, chunk_size, rngs=rngs)

        # Standard Feed-Forward Network (GLU variant is standard for DeltaNet)
        self.mlp_w1 = nnx.Linear(dim, mlp_dim, use_bias=False, rngs=rngs)
        self.mlp_w2 = nnx.Linear(dim, mlp_dim, use_bias=False, rngs=rngs)
        self.mlp_w3 = nnx.Linear(mlp_dim, dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Pre-norm architecture
        # 1. DeltaNet Sub-layer
        x = x + self.attn(self.norm1(x))
        
        # 2. MLP Sub-layer (SwiGLU)
        h = self.norm2(x)
        mlp_out = self.mlp_w3(jax.nn.silu(self.mlp_w1(h)) * self.mlp_w2(h))
        x = x + mlp_out
        
        return x

# --- 3. The Full Language Model ---
class GatedDeltaNet(nnx.Module):
    def __init__(self, vocab_size: int, num_layers: int, dim: int, num_heads: int, chunk_size: int, rngs: nnx.Rngs):
        self.embed = nnx.Embed(vocab_size, dim, rngs=rngs)
        
        # Stack multiple blocks
        self.blocks = nnx.Sequential(
            *[
                GatedDeltaNetBlock(dim, num_heads, mlp_dim=dim * 4, chunk_size=chunk_size, rngs=rngs) 
                for _ in range(num_layers)
            ]
        )
        
        self.norm_f = nnx.RMSNorm(dim, rngs=rngs)
        self.lm_head = nnx.Linear(dim, vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        x = self.embed(input_ids)
        
        x = self.blocks(x)
            
        x = self.norm_f(x)
        return self.lm_head(x)
    

# Setup Hyperparameters
rngs = nnx.Rngs(0)
batch_size, seq_len = 2, 512
vocab_size = 10000

# Instantiate the model
model = GatedDeltaNet(
    vocab_size=vocab_size, 
    num_layers=4, 
    dim=512, 
    num_heads=8, 
    chunk_size=64, # Your chunk size goes here!
    rngs=rngs
)

# Dummy text inputs (Batch, Seq)
dummy_inputs = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

# Forward Pass
logits = model(dummy_inputs)
print(f"Output shape: {logits.shape}") # Should be (2, 512, 10000)
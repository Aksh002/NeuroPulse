# Neural computation checks

The manual engine in `src/neuropulse/nn_numpy` differentiates its own dense,
activation, convolution, recurrent, and loss operations. NumPy supplies array
arithmetic only. Float64 central differences and matched PyTorch float64
computations are independent test oracles; training uses a single loss reduction.

For a dense minibatch, `Z = XW + b`, with `X:[B,D]`, `W:[D,H]`, and upstream
`G = dL/dZ:[B,H]`. The chain rule gives `dW = XᵀG`, `db = sum_B G`, and
`dX = GWᵀ`. For mean squared error over `M` scalar outputs,
`L = sum((prediction−target)²)/M` and `dL/dprediction = 2(prediction−target)/M`.
The layer never divides this gradient by batch size again. ReLU uses zero
derivative at zero; gradient tests avoid that nondifferentiable point. Sigmoid
uses a sign-aware implementation to avoid exponential overflow. BCE with
logits uses `mean(max(z,0)−z*y+log1p(exp(−abs(z))))` and gradient
`(sigmoid(z)−y)/M`.

For Conv1D cross-correlation, `Y[b,o,t] = bias[o] + Σ(c,j)
K[o,c,j] X_padded[b,c,t*s+j]`. Each output contributes
`G[b,o,t] X_padded[b,c,t*s+j]` to `dK[o,c,j]`, `G[b,o,t] K[o,c,j]` to the
corresponding padded input gradient, and `G[b,o,t]` to `db[o]`. Contributions
at overlapping input positions sum. Remove padding after accumulation.
Finite-difference and PyTorch oracle checks cover valid, padded, and strided
cases; channel and time axes are converted explicitly for PyTorch.

For the basic tanh recurrent cell,
`a_t = x_t W_x + h_(t−1) W_h + b`, `h_t = tanh(a_t)`. Reverse time combines
the output upstream gradient with the future hidden-state gradient before
forming `da_t = dh_t * (1−h_t²)`. At each time step, accumulate
`dW_x += x_tᵀ da_t`, `dW_h += h_(t−1)ᵀ da_t`, `db += sum_B da_t`, and propagate
`dh_(t−1) = da_t W_hᵀ`. Each independent window begins with zero hidden state.
The manual LSTM remains stretch scope; the production LSTM uses PyTorch.

SGD subtracts `ηg`; momentum uses `v←μv+g`, `θ←θ−ηv`; Adam uses first and
second moments with minibatch-step bias correction. Global-norm clipping scales
all parameter gradients by `min(1,c/(||g||₂+1e−12))` before the optimizer step.
Checkpoints store named numeric arrays in NPZ and architecture/state metadata
in JSON. Loads use `allow_pickle=False` and reject mismatched names or shapes.

The NASA terminal score uses `d = prediction−truth`, summing
`exp(−d/13)−1` when `d<0`, and `exp(d/10)−1` otherwise. Metrics use one
uncapped official test label per engine. The piecewise training target
`min(last_cycle−current_cycle,125)` is a modeling assumption and never clips
official truth. [The measured results](results.md) separate validation,
official terminal, and exploratory studies.

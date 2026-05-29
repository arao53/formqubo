# formqubo

Hybrid quantum-classical optimization: convert Pyomo MIP/MIQP models to QUBO for D-Wave quantum annealing, with classical polishing via IPOPT/GLPK/CBC.

## Setup with uv

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create and activate a virtual environment

```bash
uv venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### 3. Install core dependencies

```bash
uv pip install -e .
```

This installs `pyomo` and `dimod` (the required packages for simulation via simulated annealing).

### 4. Install D-Wave hardware support (optional)

To solve on real D-Wave hardware or use `LeapHybridSampler`, install the `dwave` extra:

```bash
uv pip install -e ".[dwave]"
```

You will also need a D-Wave Leap account and API token — set it via:

```bash
export DWAVE_API_TOKEN=your_token_here
```

### 5. Install a classical solver for the polishing stage

`hybridsolve.py` uses a classical solver (IPOPT, GLPK, or CBC) to polish continuous variables after the quantum stage. Install at least one:

```bash
# GLPK (open source, handles LP/MIP)
conda install -c conda-forge glpk
# or on macOS:
brew install glpk

# IPOPT (open source, handles nonlinear)
conda install -c conda-forge ipopt

# CBC (open source, handles MIP)
conda install -c conda-forge coin-or-cbc
```

## Usage

```python
import pyomo.environ as pyo
from scripts.pyomo_to_qubo import PyomoToQUBO
from scripts.hybridsolve import HybridSolver

# Build a Pyomo model
model = pyo.ConcreteModel()
# ... define variables, constraints, objective ...

# Convert to QUBO
converter = PyomoToQUBO(model)
qubo, offset = converter.to_qubo()

# Or solve end-to-end with the hybrid pipeline
solver = HybridSolver()
result = solver.solve(model)
```

## Dependencies

| Package | Role | Required |
|---|---|---|
| `pyomo` | Optimization modeling | Yes |
| `dimod` | QUBO/BQM representation + simulated annealing | Yes |
| `dwave-ocean-sdk` | D-Wave hardware samplers (`LeapHybridSampler`) | Optional |
| IPOPT / GLPK / CBC | Classical polishing solver | One required for `HybridSolver` |

"""
hybrid_pipeline.py
==================
Created by Claude code - Sonnet 4.6.
Hybrid quantum-classical optimization pipeline:

    Stage 1  [QUBO]       Convert Pyomo MIP → QUBO via pyomo_to_qubo.py
    Stage 2  [Quantum]    Solve QUBO with D-Wave (or simulated annealing locally)
    Stage 3  [Fix]        Fix binary/integer decisions in the original Pyomo model
    Stage 4  [Polish]     Solve the remaining continuous subproblem with IPOPT/GLPK

The intuition:
    - Quantum annealing is good at finding high-quality binary assignments quickly.
    - Once binaries are fixed, the remaining problem is often a continuous LP/NLP
    that classical solvers handle exactly and cheaply.
    - This hybrid often beats either approach alone on MIQPs.

Usage:
    from hybrid_pipeline import HybridPipeline

    pipeline = HybridPipeline(model, penalty=5000, use_dwave=True)
    result = pipeline.run()
    print(result.summary())

    # Or run stages manually:
    pipeline.stage1_convert()
    pipeline.stage2_solve_qubo()
    pipeline.stage3_fix_binaries()
    pipeline.stage4_polish()
    print(pipeline.result.summary())
"""

import time
import warnings
import copy
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List

import pyomo.environ as pyo
from pyomo.core import value
from pyomo.opt import SolverStatus, TerminationCondition
import dimod

from pyomo_to_qubo import PyomoToQUBO, extract_bilinear_terms


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Holds outputs and diagnostics from each pipeline stage."""

    # Stage 1
    n_pyomo_vars: int = 0
    n_qubo_vars: int = 0
    n_qubo_interactions: int = 0
    stage1_time: float = 0.0

    # Stage 2
    qubo_energy: float = float('inf')
    qubo_solution: Dict[str, float] = field(default_factory=dict)
    n_dwave_reads: int = 0
    stage2_time: float = 0.0
    sampler_used: str = ''

    # Stage 3
    binary_assignments: Dict[str, float] = field(default_factory=dict)
    stage3_time: float = 0.0

    # Stage 4
    polish_status: str = ''
    polish_objective: float = float('inf')
    polish_solution: Dict[str, float] = field(default_factory=dict)
    polish_solver: str = ''
    stage4_time: float = 0.0
    polish_feasible: bool = False

    # Overall
    total_time: float = 0.0
    pipeline_feasible: bool = False

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  Hybrid Pipeline Result",
            "=" * 60,
            f"  Stage 1  [Convert]   {self.n_pyomo_vars} Pyomo vars → "
            f"{self.n_qubo_vars} QUBO vars  ({self.stage1_time:.2f}s)",
            f"  Stage 2  [Quantum]   energy={self.qubo_energy:.4f}  "
            f"sampler={self.sampler_used}  ({self.stage2_time:.2f}s)",
            f"  Stage 3  [Fix]       {len(self.binary_assignments)} binaries fixed"
            f"  ({self.stage3_time:.2f}s)",
            f"  Stage 4  [Polish]    obj={self.polish_objective:.6f}  "
            f"status={self.polish_status}  solver={self.polish_solver}"
            f"  ({self.stage4_time:.2f}s)",
            f"  {'─'*50}",
            f"  Total time : {self.total_time:.2f}s",
            f"  Feasible   : {self.pipeline_feasible}",
            "=" * 60,
        ]
        if self.polish_solution:
            lines.append("  Final variable values:")
            for k, v in sorted(self.polish_solution.items()):
                lines.append(f"    {k:30s} = {v:.6g}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Polishing subproblem builder
# ---------------------------------------------------------------------------

def _build_fixed_subproblem(original_model: pyo.ConcreteModel,
                             binary_assignments: Dict[str, float]
                             ) -> pyo.ConcreteModel:
    """
    Clone the original Pyomo model and fix all binary/integer variables
    to their quantum-assigned values.

    Returns a new ConcreteModel where:
      - Binary/integer vars are fixed (lb == ub == assigned value)
      - Continuous vars remain free (this is what the polisher optimizes)
      - All original constraints and objective are preserved
    """
    # Clone via deepcopy so we never mutate the original
    sub = copy.deepcopy(original_model)

    fixed_count = 0
    for v in sub.component_objects(pyo.Var, active=True):
        for idx in v:
            var = v[idx] if idx is not None else v
            name = var.name

            if name in binary_assignments:
                val = binary_assignments[name]
                var.fix(val)
                fixed_count += 1

    print(f"  Fixed {fixed_count} binary/integer variables in subproblem.")
    return sub


def _unfix_continuous(model: pyo.ConcreteModel):
    """Unfix any continuous variables that were inadvertently fixed."""
    for v in model.component_objects(pyo.Var, active=True):
        for idx in v:
            var = v[idx] if idx is not None else v
            if not (var.is_binary() or var.is_integer()):
                if var.is_fixed():
                    var.unfix()


# ---------------------------------------------------------------------------
# Solver detection
# ---------------------------------------------------------------------------

def _find_solver(preferred: Optional[str] = None) -> Optional[str]:
    """
    Find an available continuous/convex solver.
    Preference order: ipopt > glpk > cbc > cplex > gurobi
    """
    candidates = ([preferred] if preferred else []) + [
        'ipopt', 'glpk', 'cbc', 'cplex', 'gurobi'
    ]
    for name in candidates:
        if name is None:
            continue
        try:
            s = pyo.SolverFactory(name)
            if s.available():
                return name
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class HybridPipeline:
    """
    Hybrid quantum-classical optimization pipeline.

    Parameters
    ----------
    model : pyo.ConcreteModel
        Original Pyomo MIP/MIQP. Must have bounds on all variables.
    penalty : float, optional
        QUBO constraint penalty. Auto-set if None.
    continuous_bits : int
        Bits for encoding continuous variables (default 6).
        Fewer bits = fewer qubits = faster quantum solve, less precise.
    bilinear_strategy : str
        'expand', 'mccormick', or 'auto' (default).
    use_dwave : bool
        If True, use D-Wave LeapHybridSampler.
        If False, use SimulatedAnnealingSampler (no account needed).
    dwave_reads : int
        Number of reads for raw QPU (ignored by hybrid sampler).
    polish_solver : str, optional
        Classical solver for Stage 4. Auto-detected if None.
    polish_solver_options : dict, optional
        Keyword options passed to the polish solver.
    integer_fix_strategy : str
        How to fix integer (non-binary) variables from QUBO solution:
        'round'  – round to nearest integer (default)
        'floor'  – floor
        'ceil'   – ceiling
    warm_start_from_qubo : bool
        If True, initialize continuous variables in the polishing subproblem
        from the QUBO decoded solution before solving (helps NLP convergence).
    """

    def __init__(
        self,
        model: pyo.ConcreteModel,
        penalty: float = None,
        continuous_bits: int = 6,
        bilinear_strategy: str = 'auto',
        use_dwave: bool = False,
        dwave_reads: int = 1000,
        polish_solver: str = None,
        polish_solver_options: dict = None,
        integer_fix_strategy: str = 'round',
        warm_start_from_qubo: bool = True,
    ):
        self.model = model
        self.penalty = penalty
        self.continuous_bits = continuous_bits
        self.bilinear_strategy = bilinear_strategy
        self.use_dwave = use_dwave
        self.dwave_reads = dwave_reads
        self.polish_solver = polish_solver
        self.polish_solver_options = polish_solver_options or {}
        self.integer_fix_strategy = integer_fix_strategy
        self.warm_start_from_qubo = warm_start_from_qubo

        # Internal state
        self._converter: Optional[PyomoToQUBO] = None
        self._bqm: Optional[dimod.BinaryQuadraticModel] = None
        self._sampleset: Optional[dimod.SampleSet] = None
        self._qubo_solution: Dict[str, float] = {}
        self._subproblem: Optional[pyo.ConcreteModel] = None
        self.result = PipelineResult()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> PipelineResult:
        """Run all four stages end-to-end."""
        t0 = time.time()
        self.stage1_convert()
        self.stage2_solve_qubo()
        self.stage3_fix_binaries()
        self.stage4_polish()
        self.result.total_time = time.time() - t0
        return self.result

    def stage1_convert(self) -> dimod.BinaryQuadraticModel:
        """Convert the Pyomo model to a QUBO BQM."""
        print("\n" + "━" * 60)
        print("  STAGE 1: Pyomo → QUBO conversion")
        print("━" * 60)
        t0 = time.time()

        self._converter = PyomoToQUBO(
            self.model,
            penalty=self.penalty,
            continuous_bits=self.continuous_bits,
            bilinear_strategy=self.bilinear_strategy,
        )
        self._bqm = self._converter.convert()

        self.result.n_pyomo_vars = self._converter._count_pyomo_vars()
        self.result.n_qubo_vars = len(self._bqm.variables)
        self.result.n_qubo_interactions = len(self._bqm.quadratic)
        self.result.stage1_time = time.time() - t0
        return self._bqm

    def stage2_solve_qubo(self) -> dimod.SampleSet:
        """Solve the QUBO using D-Wave or simulated annealing."""
        if self._bqm is None:
            raise RuntimeError("Call stage1_convert() first.")

        print("\n" + "━" * 60)
        print("  STAGE 2: QUBO solve (quantum / simulated annealing)")
        print("━" * 60)
        t0 = time.time()

        sampler, sampler_name = self._get_sampler()
        self.result.sampler_used = sampler_name
        print(f"  Sampler: {sampler_name}")

        sample_kwargs = {}
        if 'SimulatedAnnealing' in sampler_name:
            sample_kwargs['num_reads'] = self.dwave_reads
        elif 'QPU' in sampler_name or 'Embedding' in sampler_name:
            sample_kwargs['num_reads'] = self.dwave_reads

        self._sampleset = sampler.sample(self._bqm, **sample_kwargs)

        best = self._sampleset.first
        self.result.qubo_energy = best.energy
        self.result.n_dwave_reads = len(self._sampleset)
        self.result.stage2_time = time.time() - t0

        print(f"  Best energy   : {best.energy:.4f}")
        print(f"  Num samples   : {len(self._sampleset)}")

        # Decode back to Pyomo variable values
        self._qubo_solution = self._converter.decode(self._sampleset)
        self.result.qubo_solution = dict(self._qubo_solution)
        return self._sampleset

    def stage3_fix_binaries(self) -> Dict[str, float]:
        """
        Extract binary/integer decisions from the QUBO solution.
        Applies integer_fix_strategy to round non-binary integers.
        Returns the dict of fixed assignments.
        """
        if not self._qubo_solution:
            raise RuntimeError("Call stage2_solve_qubo() first.")

        print("\n" + "━" * 60)
        print("  STAGE 3: Fix binary/integer variables")
        print("━" * 60)
        t0 = time.time()

        assignments = {}
        for v in self.model.component_objects(pyo.Var, active=True):
            for idx in v:
                var = v[idx] if idx is not None else v
                name = var.name
                raw = self._qubo_solution.get(name)
                if raw is None:
                    continue

                if var.is_binary() or var.domain is pyo.Binary:
                    # Binary: threshold at 0.5
                    assignments[name] = 1.0 if raw >= 0.5 else 0.0

                elif var.is_integer() or var.domain in (
                        pyo.Integers, pyo.NonNegativeIntegers, pyo.PositiveIntegers):
                    # Integer: apply chosen rounding strategy
                    if self.integer_fix_strategy == 'floor':
                        assignments[name] = float(int(raw))
                    elif self.integer_fix_strategy == 'ceil':
                        import math
                        assignments[name] = float(math.ceil(raw))
                    else:  # 'round'
                        assignments[name] = float(round(raw))
                # Continuous vars are NOT fixed here — left for Stage 4

        self.result.binary_assignments = assignments
        self.result.stage3_time = time.time() - t0

        print(f"  Binary/integer assignments:")
        for name, val in sorted(assignments.items()):
            print(f"    {name:35s} = {val:.0f}")

        return assignments

    def stage4_polish(self) -> PipelineResult:
        """
        Fix binaries in the original model and solve the remaining
        continuous subproblem with a classical solver (IPOPT/GLPK/CBC).
        """
        if not self.result.binary_assignments:
            raise RuntimeError("Call stage3_fix_binaries() first.")

        print("\n" + "━" * 60)
        print("  STAGE 4: Polish with classical continuous solver")
        print("━" * 60)
        t0 = time.time()

        # Find solver
        solver_name = _find_solver(self.polish_solver)
        if solver_name is None:
            warnings.warn(
                "No classical solver found (tried ipopt, glpk, cbc). "
                "Install one: `conda install -c conda-forge ipopt glpk` "
                "or `pip install glpk`. Skipping Stage 4.")
            self.result.polish_status = 'no_solver'
            self.result.stage4_time = time.time() - t0
            # Fall back to QUBO solution as best available
            self.result.polish_solution = self._qubo_solution
            self.result.polish_objective = self.result.qubo_energy
            return self.result

        self.result.polish_solver = solver_name
        print(f"  Solver: {solver_name}")

        # Build fixed subproblem
        self._subproblem = _build_fixed_subproblem(
            self.model, self.result.binary_assignments)

        # Warm-start continuous variables from QUBO solution
        if self.warm_start_from_qubo:
            self._warm_start_continuous(self._subproblem, self._qubo_solution)

        # Solve
        solver = pyo.SolverFactory(solver_name)
        for k, v in self.polish_solver_options.items():
            solver.options[k] = v

        # IPOPT-specific: suppress output unless debugging
        if solver_name == 'ipopt' and 'print_level' not in self.polish_solver_options:
            solver.options['print_level'] = 3

        try:
            sol = solver.solve(self._subproblem, tee=False)
            status = sol.solver.termination_condition

            self.result.polish_status = str(status)
            self.result.polish_feasible = status in (
                TerminationCondition.optimal,
                TerminationCondition.locallyOptimal,
                TerminationCondition.feasible,
            )

            if self.result.polish_feasible:
                # Extract objective value
                for obj in self._subproblem.component_objects(
                        pyo.Objective, active=True):
                    try:
                        self.result.polish_objective = value(obj)
                    except Exception:
                        pass
                    break

                # Extract all variable values (merge binary + continuous)
                sol_dict = {}
                for v in self._subproblem.component_objects(pyo.Var, active=True):
                    for idx in v:
                        var = v[idx] if idx is not None else v
                        try:
                            sol_dict[var.name] = value(var)
                        except Exception:
                            pass
                self.result.polish_solution = sol_dict
                self.result.pipeline_feasible = True

                print(f"  Status    : {status} ✓")
                print(f"  Objective : {self.result.polish_objective:.6f}")

            else:
                print(f"  Status    : {status} — polishing infeasible/failed.")
                print("  Falling back to QUBO solution.")
                self.result.polish_solution = self._qubo_solution
                self._attempt_feasibility_recovery()

        except Exception as e:
            warnings.warn(f"Polishing solver raised exception: {e}")
            self.result.polish_status = f'error: {e}'
            self.result.polish_solution = self._qubo_solution

        self.result.stage4_time = time.time() - t0
        return self.result

    # ------------------------------------------------------------------
    # Multi-start: run Stage 2–4 with k best QUBO samples
    # ------------------------------------------------------------------

    def run_multistart(self, n_starts: int = 3) -> PipelineResult:
        """
        Run Stage 1 once, then try the top n_starts QUBO samples as
        binary seeds for polishing. Returns the best feasible result.

        Useful when the penalty landscape is tricky and a single QUBO
        sample might be a poor binary assignment.
        """
        print(f"\n{'━'*60}")
        print(f"  MULTISTART: trying top {n_starts} QUBO samples")
        print(f"{'━'*60}")

        if self._bqm is None:
            self.stage1_convert()
        if self._sampleset is None:
            self.stage2_solve_qubo()

        best_result = None
        best_obj = float('inf')

        samples = list(self._sampleset.samples())[:n_starts]
        energies = list(self._sampleset.data_vectors['energy'])[:n_starts]

        for i, (sample, energy) in enumerate(zip(samples, energies)):
            print(f"\n  --- Start {i+1}/{n_starts}  (QUBO energy={energy:.4f}) ---")

            # Temporarily override QUBO solution with this sample
            decoded = self._decode_sample(sample)
            self._qubo_solution = decoded
            self.result.qubo_solution = decoded
            self.result.qubo_energy = energy

            self.stage3_fix_binaries()
            self.stage4_polish()

            if (self.result.polish_feasible and
                    self.result.polish_objective < best_obj):
                best_obj = self.result.polish_objective
                best_result = copy.deepcopy(self.result)
                print(f"  ✓ New best: {best_obj:.6f}")

        if best_result:
            self.result = best_result
            print(f"\n  Best multistart objective: {best_obj:.6f}")
        else:
            print("\n  No feasible multistart solution found.")

        return self.result

    # ------------------------------------------------------------------
    # Feasibility recovery: try relaxing binary assignments one by one
    # ------------------------------------------------------------------

    def _attempt_feasibility_recovery(self, max_flips: int = 5):
        """
        If polishing is infeasible, try flipping the most uncertain
        binary variables (those closest to 0.5 in the QUBO solution)
        and re-solving. A simple but often effective repair heuristic.
        """
        print(f"\n  Attempting feasibility recovery (up to {max_flips} flips)...")

        # Rank binaries by proximity to 0.5 (most uncertain)
        uncertain = []
        for name, val in self._qubo_solution.items():
            var = self._get_pyomo_var(name)
            if var is not None and (var.is_binary() or var.domain is pyo.Binary):
                uncertain.append((abs(val - 0.5), name, val))
        uncertain.sort()  # smallest distance to 0.5 = most uncertain

        assignments = dict(self.result.binary_assignments)

        for _, name, _ in uncertain[:max_flips]:
            # Flip this binary
            assignments[name] = 1.0 - assignments[name]
            print(f"    Flipping {name} → {assignments[name]:.0f}")

            sub = _build_fixed_subproblem(self.model, assignments)
            if self.warm_start_from_qubo:
                self._warm_start_continuous(sub, self._qubo_solution)

            solver = pyo.SolverFactory(self.result.polish_solver)
            if self.result.polish_solver == 'ipopt':
                solver.options['print_level'] = 0
            try:
                sol = solver.solve(sub, tee=False)
                status = sol.solver.termination_condition
                if status in (TerminationCondition.optimal,
                               TerminationCondition.locallyOptimal,
                               TerminationCondition.feasible):
                    obj = None
                    for o in sub.component_objects(pyo.Objective, active=True):
                        obj = value(o)
                        break
                    print(f"    ✓ Feasible after flip! obj={obj:.6f}")
                    self.result.polish_feasible = True
                    self.result.polish_objective = obj
                    self.result.binary_assignments = assignments
                    sol_dict = {}
                    for v in sub.component_objects(pyo.Var, active=True):
                        for idx in v:
                            var = v[idx] if idx is not None else v
                            try:
                                sol_dict[var.name] = value(var)
                            except Exception:
                                pass
                    self.result.polish_solution = sol_dict
                    self.result.pipeline_feasible = True
                    return
            except Exception:
                pass

        print("    Recovery unsuccessful.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_sampler(self):
        if self.use_dwave:
            try:
                from dwave.system import LeapHybridSampler
                return LeapHybridSampler(), 'LeapHybridSampler'
            except Exception as e:
                warnings.warn(f"D-Wave unavailable ({e}); falling back to SA.")

        sa = dimod.SimulatedAnnealingSampler()
        return sa, 'SimulatedAnnealingSampler'

    def _warm_start_continuous(self, model: pyo.ConcreteModel,
                                solution: Dict[str, float]):
        """Set initial values for continuous variables from a solution dict."""
        count = 0
        for v in model.component_objects(pyo.Var, active=True):
            for idx in v:
                var = v[idx] if idx is not None else v
                if var.is_fixed():
                    continue
                name = var.name
                if name in solution:
                    try:
                        var.set_value(solution[name])
                        count += 1
                    except Exception:
                        pass
        print(f"  Warm-started {count} continuous variables from QUBO solution.")

    def _decode_sample(self, sample: dict) -> Dict[str, float]:
        """Decode a raw QUBO sample dict back to Pyomo variable values."""
        # Temporarily patch the converter's sampleset with a fake single sample
        fake_ss = dimod.SampleSet.from_samples(
            sample, vartype='BINARY', energy=0.0)
        return self._converter.decode(fake_ss)

    def _get_pyomo_var(self, name: str):
        """Look up a Pyomo variable by name."""
        for v in self.model.component_objects(pyo.Var, active=True):
            for idx in v:
                var = v[idx] if idx is not None else v
                if var.name == name:
                    return var
        return None


# ---------------------------------------------------------------------------
# Example: energy scheduling MIQP with bilinear revenue term
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import dimod

    # -----------------------------------------------------------------------
    # Build a unit commitment model with bilinear revenue (price * output)
    # -----------------------------------------------------------------------
    m = pyo.ConcreteModel()

    G = [0, 1, 2]   # generators
    T = [0, 1, 2]   # time periods

    cost   = {0: 10,  1: 25,  2: 18}   # $/MWh operating cost
    pmin   = {0: 20,  1: 10,  2: 30}   # MW minimum output when on
    pmax   = {0: 100, 1: 80,  2: 120}  # MW maximum output
    demand = {0: 80,  1: 120, 2: 100}  # MW demand per period
    startup_cost = {0: 50, 1: 30, 2: 80}  # $ startup cost

    # Binary: unit g on at time t
    m.u = pyo.Var(G, T, domain=pyo.Binary)

    # Binary: unit g starts up at time t
    m.v = pyo.Var(G, T, domain=pyo.Binary)

    # Continuous: power output (MW)
    m.p = pyo.Var(G, T, domain=pyo.NonNegativeReals,
                  bounds=lambda m, g, t: (0, pmax[g]))

    # Objective: minimize total operating + startup cost
    # Note: p[g,t] * u[g,t] is bilinear (continuous * binary)
    m.obj = pyo.Objective(
        expr=(
            sum(cost[g] * m.p[g, t] for g in G for t in T) +
            sum(startup_cost[g] * m.v[g, t] for g in G for t in T)
        ),
        sense=pyo.minimize
    )

    # Demand satisfaction
    m.demand_con = pyo.Constraint(T, rule=lambda m, t:
        sum(m.p[g, t] for g in G) >= demand[t])

    # Output <= pmax when on, 0 when off
    m.pmax_con = pyo.Constraint(G, T, rule=lambda m, g, t:
        m.p[g, t] <= pmax[g] * m.u[g, t])   # bilinear: continuous * binary

    # Output >= pmin when on
    m.pmin_con = pyo.Constraint(G, T, rule=lambda m, g, t:
        m.p[g, t] >= pmin[g] * m.u[g, t])   # bilinear: continuous * binary

    # Startup logic: v[g,t] >= u[g,t] - u[g,t-1]
    m.startup_con = pyo.Constraint(G, T, rule=lambda m, g, t:
        m.v[g, t] >= m.u[g, t] - (m.u[g, t-1] if t > 0 else 0))

    print("=" * 60)
    print("  Hybrid Quantum-Classical Pipeline Demo")
    print("  Unit Commitment with Bilinear Terms")
    print("=" * 60)
    print(f"  Generators : {len(G)}")
    print(f"  Time steps : {len(T)}")
    print(f"  Variables  : {len(G)*len(T)} binary (u,v) + {len(G)*len(T)} continuous (p)")

    # -----------------------------------------------------------------------
    # Run the pipeline
    # -----------------------------------------------------------------------
    pipeline = HybridPipeline(
        model=m,
        penalty=3000,
        continuous_bits=5,        # keep QUBO small for demo
        bilinear_strategy='auto',
        use_dwave=False,           # set True if you have a D-Wave account
        dwave_reads=500,
        polish_solver=None,        # auto-detect (ipopt > glpk > cbc)
        integer_fix_strategy='round',
        warm_start_from_qubo=True,
    )

    # --- Option A: single run ---
    result = pipeline.run()
    print(result.summary())

    # --- Option B: multistart (uncomment to try) ---
    # result = pipeline.run_multistart(n_starts=3)
    # print(result.summary())
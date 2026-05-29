"""
pyomo_to_qubo.py
================
Converts a Pyomo MIP / MIQP to a QUBO formulation for D-Wave's Ocean SDK.

Supports:
    - Binary variables (direct mapping)
    - Integer variables (binary encoding)
    - Continuous variables (discretized binary encoding)
    - Linear objectives and constraints
    - Quadratic / bilinear objectives and constraints via two strategies:
      * 'expand'     : exact cross-product of binary encodings  (more qubits)
      * 'mccormick'  : McCormick envelope linearization          (approximate, fewer qubits)

Usage:
    from pyomo_to_qubo import PyomoToQUBO
    from dwave.system import LeapHybridSampler

    converter = PyomoToQUBO(model, penalty=10.0, bilinear_strategy='mccormick')
    bqm = converter.convert()
    result = converter.sample(bqm)
    solution = converter.decode(result)
"""

import math
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import pyomo.environ as pyo
from pyomo.core import value
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.visitor import ExpressionValueVisitor
import dimod


# ---------------------------------------------------------------------------
# Bilinear term extraction
# ---------------------------------------------------------------------------

def extract_bilinear_terms(expr):
    """
    Walk a Pyomo expression tree and extract:
        - constant
        - linear_terms  : list of (var, coef)
        - bilinear_terms: list of (var1, var2, coef)   [includes x^2 as (x,x,c)]

    Returns (constant, linear_terms, bilinear_terms).
    Works for ProductExpr, SumExpr, MonomialTermExpr, LinearExpr, etc.
    Falls back to generate_standard_repn for linear parts.
    """
    from pyomo.core.expr.numeric_expr import (
        ProductExpression, SumExpression, MonomialTermExpression,
        LinearExpression, NegationExpression, PowExpression,
        DivisionExpression,
    )

    constant = 0.0
    linear = []      # (var, coef)
    bilinear = []    # (var1, var2, coef)

    def _walk(node, multiplier=1.0):
        nonlocal constant

        if node is None:
            return
        if isinstance(node, (int, float)):
            constant += multiplier * node
            return

        # Numeric constant leaf
        try:
            v = value(node, exception=False)
            if v is not None and not node.is_variable_type():
                constant += multiplier * float(v)
                return
        except Exception:
            pass

        # Variable leaf
        if node.is_variable_type():
            linear.append((node, multiplier))
            return

        # Monomial: coef * var
        if isinstance(node, MonomialTermExpression):
            coef_node, var_node = node.args
            try:
                c = float(value(coef_node))
            except Exception:
                c = 1.0
            if var_node.is_variable_type():
                linear.append((var_node, multiplier * c))
            else:
                _walk(var_node, multiplier * c)
            return

        # Sum
        if isinstance(node, (SumExpression, LinearExpression)):
            for child in node.args:
                _walk(child, multiplier)
            return

        # Negation
        if isinstance(node, NegationExpression):
            _walk(node.args[0], -multiplier)
            return

        # Product: may be bilinear
        if isinstance(node, ProductExpression):
            left, right = node.args
            left_vars = list(identify_variables(left))
            right_vars = list(identify_variables(right))

            if not left_vars and not right_vars:
                # constant * constant
                try:
                    constant += multiplier * float(value(left)) * float(value(right))
                except Exception:
                    pass
                return

            if not left_vars:
                # scalar * expr
                try:
                    c = float(value(left))
                    _walk(right, multiplier * c)
                except Exception:
                    warnings.warn(f"Could not evaluate left side of product; skipping.")
                return

            if not right_vars:
                try:
                    c = float(value(right))
                    _walk(left, multiplier * c)
                except Exception:
                    warnings.warn(f"Could not evaluate right side of product; skipping.")
                return

            # Both sides have variables — bilinear
            # Flatten each side to (var, coef) pairs
            left_terms = _flatten_to_linear(left)
            right_terms = _flatten_to_linear(right)

            for (v1, c1) in left_terms:
                for (v2, c2) in right_terms:
                    bilinear.append((v1, v2, multiplier * c1 * c2))
            return

        # Power: x^2 treated as x*x
        if isinstance(node, PowExpression):
            base, exp_node = node.args
            try:
                exp = int(value(exp_node))
            except Exception:
                warnings.warn(f"Non-integer exponent in expression; skipping.")
                return
            if exp == 1:
                _walk(base, multiplier)
            elif exp == 2:
                base_vars = list(identify_variables(base))
                if len(base_vars) == 1 and base_vars[0].is_variable_type():
                    bilinear.append((base_vars[0], base_vars[0], multiplier))
                else:
                    warnings.warn(f"Squared non-variable expression; skipping.")
            else:
                warnings.warn(f"Exponent {exp} > 2 not supported; skipping.")
            return

        # Division: treat as scalar if denominator is constant
        if isinstance(node, DivisionExpression):
            num, den = node.args
            den_vars = list(identify_variables(den))
            if not den_vars:
                try:
                    c = float(value(den))
                    _walk(num, multiplier / c)
                    return
                except Exception:
                    pass
            warnings.warn("Division by variable expression not supported; skipping.")
            return

        # Fallback: try linear repn
        try:
            repn = generate_standard_repn(node, compute_values=False)
            if repn.constant:
                constant += multiplier * float(value(repn.constant))
            if repn.linear_vars:
                for var, coef in zip(repn.linear_vars, repn.linear_coefs):
                    linear.append((var, multiplier * float(value(coef))))
            if repn.quadratic_vars:
                for (v1, v2), coef in zip(repn.quadratic_vars, repn.quadratic_coefs):
                    bilinear.append((v1, v2, multiplier * float(value(coef))))
        except Exception as e:
            warnings.warn(f"Could not parse expression node {type(node).__name__}: {e}")

    def _flatten_to_linear(node):
        """Return [(var, coef)] for a purely linear sub-expression."""
        terms = []
        try:
            repn = generate_standard_repn(node, compute_values=False)
            if repn.linear_vars:
                for var, coef in zip(repn.linear_vars, repn.linear_coefs):
                    terms.append((var, float(value(coef))))
            elif list(identify_variables(node)):
                # single variable
                vars_ = list(identify_variables(node))
                if len(vars_) == 1:
                    terms.append((vars_[0], 1.0))
        except Exception:
            vars_ = list(identify_variables(node))
            for v in vars_:
                terms.append((v, 1.0))
        return terms if terms else []

    _walk(expr)
    return constant, linear, bilinear


# ---------------------------------------------------------------------------
# Binary variable encoding
# ---------------------------------------------------------------------------

class BinaryEncoding:
    """
    Encodes a single non-binary variable as a sum of weighted binary variables.

    For integers:   x = lb + sum_k  2^k * q_k   (binary expansion)
    For continuous: x = lb + sum_k  delta * q_k  (uniform discretization)
    """

    def __init__(self, name: str, lb: float, ub: float, domain: str,
                 n_bits: int = None):
        self.name = name
        self.lb = lb
        self.ub = ub
        self.domain = domain
        self.range = ub - lb

        if domain == 'integer':
            n_values = int(round(self.range)) + 1
            self.n_bits = math.ceil(math.log2(max(n_values, 2)))
            self.weights = [2 ** k for k in range(self.n_bits)]
            max_sum = sum(self.weights[:-1])
            self.weights[-1] = max(min(self.weights[-1], int(self.range) - max_sum), 1)
        else:
            self.n_bits = n_bits or 8
            self.weights = [
                self.range / (2 ** self.n_bits - 1) * (2 ** k)
                for k in range(self.n_bits)
            ]

        self.qubo_vars = [f"{name}_b{k}" for k in range(self.n_bits)]

    def linear_combination(self) -> Dict[str, float]:
        """Returns {qubo_var: weight} such that x ≈ lb + sum(weight * q)."""
        return dict(zip(self.qubo_vars, self.weights))


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

class PyomoToQUBO:
    """
    Converts a Pyomo ConcreteModel (MIP or MIQP with bilinear terms) to a
    dimod BinaryQuadraticModel (QUBO) for D-Wave.

    Parameters
    ----------
    model : pyo.ConcreteModel
        Pyomo model. All non-binary variables must have finite bounds.
    penalty : float, optional
        Penalty coefficient P for constraint violations.
        Auto-set to 10x max objective coefficient if not provided.
    continuous_bits : int
        Bits for discretizing continuous variables (default 8).
        More bits = better precision, more QUBO variables.
    bilinear_strategy : str
        How to handle bilinear (x*y) terms:
        - 'expand'     : exact binary cross-product (n_bits^2 interactions per term)
        - 'mccormick'  : McCormick envelope linearization (fewer variables, approximate)
        Default: 'expand' for binary*anything, 'mccormick' for continuous*continuous.
    """

    def __init__(self, model: pyo.ConcreteModel,
                penalty: float = None,
                continuous_bits: int = 8,
                bilinear_strategy: str = 'auto'):
        self.model = model
        self.continuous_bits = continuous_bits
        self.bilinear_strategy = bilinear_strategy  # 'expand', 'mccormick', 'auto'

        self._qubo: Dict[Tuple[str, str], float] = defaultdict(float)
        self._encodings: Dict[str, BinaryEncoding] = {}
        self._binary_map: Dict[str, str] = {}
        self._offset = 0.0
        self._penalty = penalty
        self._obj_magnitude = 1.0

        # McCormick auxiliary vars: (name1, name2) -> aux_var_name
        self._mccormick_aux: Dict[Tuple[str, str], str] = {}
        self._mccormick_constraints: List = []  # extra linear constraints to encode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(self) -> dimod.BinaryQuadraticModel:
        """Run the full conversion pipeline and return a dimod BQM."""
        print("[1/5] Encoding variables...")
        self._encode_variables()

        print("[2/5] Encoding objective (including bilinear terms)...")
        self._encode_objective()

        if self._penalty is None:
            self._penalty = max(self._obj_magnitude * 10, 100)
            print(f"      Auto-penalty set to {self._penalty:.1f}")

        print(f"[3/5] Encoding constraints (P={self._penalty})...")
        self._encode_constraints()

        if self._mccormick_constraints:
            print(f"[4/5] Encoding {len(self._mccormick_constraints)} "
                    f"McCormick auxiliary constraints...")
            self._encode_mccormick_constraints()
        else:
            print("[4/5] No McCormick auxiliary constraints.")

        print("[5/5] Building BQM...")
        bqm = self._build_bqm()

        print(f"\n✓ Conversion complete.")
        print(f"  Original Pyomo variables      : {self._count_pyomo_vars()}")
        print(f"  QUBO binary variables          : {len(bqm.variables)}")
        print(f"  QUBO interactions              : {len(bqm.quadratic)}")
        print(f"  McCormick aux variables        : {len(self._mccormick_aux)}")
        return bqm

    def sample(self, bqm: dimod.BinaryQuadraticModel,
               sampler=None, **kwargs) -> dimod.SampleSet:
        """Sample the BQM using D-Wave (default: LeapHybridSampler)."""
        if sampler is None:
            try:
                from dwave.system import LeapHybridSampler
                sampler = LeapHybridSampler()
                print("Using LeapHybridSampler.")
            except ImportError:
                raise ImportError("Run: pip install dwave-ocean-sdk")
        kwargs.setdefault('label', 'pyomo_to_qubo')
        return sampler.sample(bqm, **kwargs)

    def decode(self, sampleset: dimod.SampleSet) -> Dict[str, float]:
        """Decode the best sample back into original Pyomo variable values."""
        best = sampleset.first.sample
        result = {}

        for var_name, qubo_name in self._binary_map.items():
            result[var_name] = float(best.get(qubo_name, 0))

        for var_name, enc in self._encodings.items():
            if var_name.startswith('__'):
                continue  # skip internal slack / McCormick aux
            val = enc.lb
            for qv, w in enc.linear_combination().items():
                val += w * best.get(qv, 0)
            val = max(enc.lb, min(enc.ub, val))
            if enc.domain == 'integer':
                val = round(val)
            result[var_name] = val

        print(f"\nBest sample energy : {sampleset.first.energy:.4f}")
        print(f"Decoded values     : {result}")
        return result

    def check_feasibility(self, solution: Dict[str, float],
                            tol: float = 1e-3) -> bool:
        """Check whether a decoded solution satisfies all Pyomo constraints."""
        feasible = True
        for c in self.model.component_objects(pyo.Constraint, active=True):
            for idx in c:
                con = c[idx] if idx is not None else c
                body_val = self._eval_expr_numeric(con.body, solution)
                lb = value(con.lb) if con.lb is not None else None
                ub = value(con.ub) if con.ub is not None else None
                ok = True
                if lb is not None and body_val < lb - tol:
                    ok = False
                if ub is not None and body_val > ub + tol:
                    ok = False
                if not ok:
                    print(f"  VIOLATED: {c.name}[{idx}]  "
                        f"body={body_val:.4f}  lb={lb}  ub={ub}")
                    feasible = False
        if feasible:
            print("✓ All constraints satisfied.")
        return feasible

    # ------------------------------------------------------------------
    # Step 1: Variable encoding
    # ------------------------------------------------------------------

    def _encode_variables(self):
        for v in self.model.component_objects(pyo.Var, active=True):
            for idx in v:
                var = v[idx] if idx is not None else v
                name = self._var_name(var)
                domain = var.domain

                if var.is_binary() or domain is pyo.Binary:
                    self._binary_map[name] = name
                else:
                    lb, ub = self._get_bounds(var)
                    if (var.is_integer() or domain in
                            (pyo.Integers, pyo.NonNegativeIntegers, pyo.PositiveIntegers)):
                        enc = BinaryEncoding(name, lb, ub, 'integer')
                    else:
                        enc = BinaryEncoding(name, lb, ub, 'continuous',
                                            n_bits=self.continuous_bits)
                        warnings.warn(
                            f"Continuous var '{name}' discretized to "
                            f"{self.continuous_bits} bits "
                            f"(precision ≈ {(ub-lb)/(2**self.continuous_bits):.4f})")
                    self._encodings[name] = enc

    # ------------------------------------------------------------------
    # Step 2: Objective encoding
    # ------------------------------------------------------------------

    def _encode_objective(self):
        obj = None
        for o in self.model.component_objects(pyo.Objective, active=True):
            obj = o
            break
        if obj is None:
            return

        sense = 1.0 if obj.sense == pyo.minimize else -1.0
        const, linear, bilinear = extract_bilinear_terms(obj.expr)

        self._offset += sense * const

        linear_coeffs = {self._var_name(v): sense * c for v, c in linear}
        if linear_coeffs:
            self._obj_magnitude = max(abs(c) for c in linear_coeffs.values())
        self._add_linear_to_qubo(linear_coeffs)

        for v1, v2, coef in bilinear:
            self._obj_magnitude = max(self._obj_magnitude, abs(coef))
            self._encode_bilinear(self._var_name(v1), self._var_name(v2),
                                  sense * coef, context='objective')

    # ------------------------------------------------------------------
    # Step 3: Constraint encoding
    # ------------------------------------------------------------------

    def _encode_constraints(self):
        P = self._penalty
        slack_counter = [0]

        for c in self.model.component_objects(pyo.Constraint, active=True):
            for idx in c:
                con = c[idx] if idx is not None else c
                const, linear, bilinear = extract_bilinear_terms(con.body)

                linear_coeffs = {self._var_name(v): coef for v, coef in linear}

                # Handle any bilinear terms in constraints
                # Strategy: substitute each bilinear with a McCormick aux var
                bilinear_linear_coeffs = {}
                for v1, v2, coef in bilinear:
                    aux_name = self._get_mccormick_aux(
                        self._var_name(v1), self._var_name(v2))
                    bilinear_linear_coeffs[aux_name] = (
                        bilinear_linear_coeffs.get(aux_name, 0) + coef)

                all_linear = {**linear_coeffs, **bilinear_linear_coeffs}

                lb_con = value(con.lb) if con.lb is not None else None
                ub_con = value(con.ub) if con.ub is not None else None

                if lb_con is not None and ub_con is not None and abs(lb_con - ub_con) < 1e-8:
                    rhs = lb_con - const
                    self._encode_equality_penalty(all_linear, rhs, P)
                else:
                    if lb_con is not None:
                        slack_name, slack_enc = self._make_slack(
                            slack_counter,
                            lo=0,
                            hi=abs(ub_con - lb_con) if ub_con else 1e4)
                        merged = dict(self._expand_to_qubo_terms(all_linear))
                        for qv, w in slack_enc.linear_combination().items():
                            merged[qv] = merged.get(qv, 0) - w
                        self._encode_equality_penalty(
                            merged, lb_con - const - slack_enc.lb, P)

                    if ub_con is not None:
                        slack_name, slack_enc = self._make_slack(
                            slack_counter,
                            lo=0,
                            hi=abs(ub_con - (lb_con or 0)) + 1)
                        flipped = {k: -v for k, v in all_linear.items()}
                        merged = dict(self._expand_to_qubo_terms(flipped))
                        for qv, w in slack_enc.linear_combination().items():
                            merged[qv] = merged.get(qv, 0) - w
                        self._encode_equality_penalty(
                            merged, -ub_con + const - slack_enc.lb, P)

    # ------------------------------------------------------------------
    # Bilinear encoding: the core new logic
    # ------------------------------------------------------------------

    def _encode_bilinear(self, name1: str, name2: str, coef: float,
                        context: str = 'objective'):
        """
        Dispatch bilinear term coef * x * y to the right strategy.

        Cases:
          binary  * binary     -> direct QUBO interaction (always exact)
          binary  * encoded    -> distribute over bits (exact)
          encoded * encoded    -> expand or McCormick (user choice)
        """
        is_bin1 = name1 in self._binary_map
        is_bin2 = name2 in self._binary_map

        if is_bin1 and is_bin2:
            # Native QUBO: binary * binary
            q1 = self._binary_map[name1]
            q2 = self._binary_map[name2]
            if q1 == q2:
                # x^2 = x for binary
                self._qubo[(q1, q1)] += coef
            else:
                key = (min(q1, q2), max(q1, q2))
                self._qubo[key] += coef

        elif is_bin1 and name2 in self._encodings:
            # binary * encoded: distribute
            q1 = self._binary_map[name1]
            enc2 = self._encodings[name2]
            self._offset += coef * enc2.lb  # binary * lb term handled as offset * binary
            # Actually: u * (lb + sum w_k q_k) = u*lb + sum w_k*(u*q_k)
            # u*lb is linear in u:
            self._qubo[(q1, q1)] += coef * enc2.lb
            for qv2, w2 in enc2.linear_combination().items():
                key = (min(q1, qv2), max(q1, qv2))
                self._qubo[key] += coef * w2

        elif name1 in self._encodings and is_bin2:
            # symmetric
            self._encode_bilinear(name2, name1, coef, context)

        else:
            # encoded * encoded
            strategy = self.bilinear_strategy
            if strategy == 'auto':
                # Use expand if at least one is integer (fewer bits), mccormick otherwise
                enc1 = self._encodings.get(name1)
                enc2 = self._encodings.get(name2)
                if enc1 and enc2:
                    total_interactions = enc1.n_bits * enc2.n_bits
                    strategy = 'expand' if total_interactions <= 64 else 'mccormick'
                else:
                    strategy = 'expand'

            if strategy == 'expand':
                self._encode_bilinear_expand(name1, name2, coef)
            else:
                self._encode_bilinear_mccormick(name1, name2, coef)

    def _encode_bilinear_expand(self, name1: str, name2: str, coef: float):
        """
        Exact expansion: x*y = (lb1 + sum_i w_i p_i)(lb2 + sum_j v_j q_j)
        Expands to constant + linear + quadratic QUBO terms.
        """
        enc1 = self._encodings[name1]
        enc2 = self._encodings[name2]

        # constant: lb1 * lb2
        self._offset += coef * enc1.lb * enc2.lb

        # linear from enc1: coef * lb2 * w_i * p_i
        for qv1, w1 in enc1.linear_combination().items():
            self._qubo[(qv1, qv1)] += coef * enc2.lb * w1

        # linear from enc2: coef * lb1 * v_j * q_j
        for qv2, w2 in enc2.linear_combination().items():
            self._qubo[(qv2, qv2)] += coef * enc1.lb * w2

        # quadratic cross terms: coef * w_i * v_j * p_i * q_j
        for qv1, w1 in enc1.linear_combination().items():
            for qv2, w2 in enc2.linear_combination().items():
                if qv1 == qv2:
                    # same qubit: p*p = p for binary
                    self._qubo[(qv1, qv1)] += coef * w1 * w2
                else:
                    key = (min(qv1, qv2), max(qv1, qv2))
                    self._qubo[key] += coef * w1 * w2

        n_interactions = enc1.n_bits * enc2.n_bits
        print(f"    [expand] {name1} × {name2}: "
            f"{enc1.n_bits}×{enc2.n_bits} = {n_interactions} interactions")

    def _encode_bilinear_mccormick(self, name1: str, name2: str, coef: float):
        """
        McCormick envelope: introduce auxiliary variable z ≈ x*y
        and add 4 linear McCormick constraints as QUBO penalties.

        z >= xL*y + x*yL - xL*yL
        z >= xU*y + x*yU - xU*yU
        z <= xU*y + x*yL - xU*yL
        z <= xL*y + x*yU - xL*yU
        """
        aux_name = self._get_mccormick_aux(name1, name2)

        enc1 = self._encodings[name1]
        enc2 = self._encodings[name2]
        xL, xU = enc1.lb, enc1.ub
        yL, yU = enc2.lb, enc2.ub

        # Add coef * z to objective (z stands in for x*y)
        self._add_linear_to_qubo({aux_name: coef})

        # Store McCormick constraints for later encoding
        # Each is a linear inequality involving x, y, z
        # We'll encode them as penalty terms in _encode_mccormick_constraints
        self._mccormick_constraints.append({
            'aux': aux_name, 'x': name1, 'y': name2,
            'xL': xL, 'xU': xU, 'yL': yL, 'yU': yU,
        })
        print(f"    [mccormick] {name1} × {name2} → aux var '{aux_name}'")

    def _get_mccormick_aux(self, name1: str, name2: str) -> str:
        """Get or create a McCormick auxiliary variable for x*y."""
        key = (min(name1, name2), max(name1, name2))
        if key not in self._mccormick_aux:
            aux_name = f"__mc_{len(self._mccormick_aux)}"
            self._mccormick_aux[key] = aux_name
            # Bounds on z = x*y
            enc1 = self._encodings[name1]
            enc2 = self._encodings[name2]
            xL, xU = enc1.lb, enc1.ub
            yL, yU = enc2.lb, enc2.ub
            products = [xL*yL, xL*yU, xU*yL, xU*yU]
            z_lb, z_ub = min(products), max(products)
            # Register aux as a continuous encoded variable
            enc_aux = BinaryEncoding(aux_name, z_lb, z_ub, 'continuous',
                                    n_bits=self.continuous_bits)
            self._encodings[aux_name] = enc_aux
        return self._mccormick_aux[key]

    def _encode_mccormick_constraints(self):
        """
        Encode the 4 McCormick envelope inequalities for each bilinear pair
        as QUBO penalty terms.
        """
        P = self._penalty
        slack_counter = [1000]  # offset to avoid collision with main slack vars

        for mc in self._mccormick_constraints:
            z = mc['aux']
            x, y = mc['x'], mc['y']
            xL, xU = mc['xL'], mc['xU']
            yL, yU = mc['yL'], mc['yU']

            # 4 McCormick inequalities, each encoded as:
            # lhs >= 0  →  lhs - slack = 0
            constraints = [
                # z >= xL*y + x*yL - xL*yL  →  z - xL*y - x*yL + xL*yL >= 0
                {'coeffs': {z: 1.0, y: -xL, x: -yL}, 'rhs_const': xL * yL,
                'type': 'ge', 'slack_range': (0, (xU-xL)*(yU-yL)+1)},
                # z >= xU*y + x*yU - xU*yU  →  z - xU*y - x*yU + xU*yU >= 0
                {'coeffs': {z: 1.0, y: -xU, x: -yU}, 'rhs_const': xU * yU,
                'type': 'ge', 'slack_range': (0, (xU-xL)*(yU-yL)+1)},
                # z <= xU*y + x*yL - xU*yL  →  xU*y + x*yL - xU*yL - z >= 0
                {'coeffs': {z: -1.0, y: xU, x: yL}, 'rhs_const': -xU * yL,
                'type': 'ge', 'slack_range': (0, (xU-xL)*(yU-yL)+1)},
                # z <= xL*y + x*yU - xL*yU  →  xL*y + x*yU - xL*yU - z >= 0
                {'coeffs': {z: -1.0, y: xL, x: yU}, 'rhs_const': -xL * yU,
                'type': 'ge', 'slack_range': (0, (xU-xL)*(yU-yL)+1)},
            ]

            for con in constraints:
                slack_name, slack_enc = self._make_slack(
                    slack_counter, lo=con['slack_range'][0], hi=con['slack_range'][1])
                coeffs_with_slack = dict(con['coeffs'])
                # Expand all variables to QUBO terms
                expanded = dict(self._expand_to_qubo_terms(coeffs_with_slack))
                # Subtract slack bits
                for qv, w in slack_enc.linear_combination().items():
                    expanded[qv] = expanded.get(qv, 0) - w
                rhs = con['rhs_const'] - slack_enc.lb
                self._encode_equality_penalty(expanded, rhs, P)

    # ------------------------------------------------------------------
    # Equality penalty encoding
    # ------------------------------------------------------------------

    def _encode_equality_penalty(self, terms: Dict[str, float],
                                rhs: float, P: float):
        """
        Add P*(sum_i a_i q_i - rhs)^2 to QUBO.
        `terms` must already be in QUBO variable space (pre-expanded).
        """
        self._offset += P * rhs ** 2

        term_list = list(terms.items())
        for i, (q1, a1) in enumerate(term_list):
            # diagonal: P*a1^2*q1 - 2*P*rhs*a1*q1
            self._qubo[(q1, q1)] += P * a1 * a1 - 2 * P * rhs * a1
            for j in range(i + 1, len(term_list)):
                q2, a2 = term_list[j]
                key = (min(q1, q2), max(q1, q2))
                self._qubo[key] += 2 * P * a1 * a2

    # ------------------------------------------------------------------
    # QUBO helpers
    # ------------------------------------------------------------------

    def _expand_to_qubo_terms(self, coeffs: Dict[str, float]) -> Dict[str, float]:
        """
        Expand {pyomo_var_name: coef} → {qubo_var: effective_coef},
        absorbing lb offsets into self._offset.
        """
        result = {}
        for name, coef in coeffs.items():
            if name in self._binary_map:
                qv = self._binary_map[name]
                result[qv] = result.get(qv, 0) + coef
            elif name in self._encodings:
                enc = self._encodings[name]
                self._offset += coef * enc.lb
                for qv, w in enc.linear_combination().items():
                    result[qv] = result.get(qv, 0) + coef * w
            else:
                warnings.warn(f"Variable '{name}' not in encoding map; skipped.")
        return result

    def _add_linear_to_qubo(self, coeffs: Dict[str, float]):
        terms = self._expand_to_qubo_terms(coeffs)
        for qv, coef in terms.items():
            self._qubo[(qv, qv)] += coef

    def _build_bqm(self) -> dimod.BinaryQuadraticModel:
        bqm = dimod.BinaryQuadraticModel('BINARY')
        for (u, v), bias in self._qubo.items():
            if abs(bias) < 1e-10:
                continue
            if u == v:
                bqm.add_variable(u, bias)
            else:
                bqm.add_interaction(u, v, bias)
        bqm.offset += self._offset
        return bqm

    def _make_slack(self, counter, lo=0, hi=100):
        counter[0] += 1
        lo = max(lo, 0)
        hi = max(hi, lo + 1)
        name = f"__slack_{counter[0]}"
        enc = BinaryEncoding(name, float(lo), float(hi), 'integer')
        self._encodings[name] = enc
        return name, enc

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _var_name(self, var) -> str:
        return var.name

    def _count_pyomo_vars(self) -> int:
        return sum(len(list(v))
                for v in self.model.component_objects(pyo.Var, active=True))

    def _get_bounds(self, var) -> Tuple[float, float]:
        lb = value(var.lb) if var.lb is not None else None
        ub = value(var.ub) if var.ub is not None else None
        if lb is None or ub is None:
            raise ValueError(
                f"Variable '{var.name}' needs finite bounds. "
                f"Got lb={lb}, ub={ub}. Use var.setlb() / var.setub().")
        return float(lb), float(ub)

    def _eval_expr_numeric(self, expr, solution: Dict[str, float]) -> float:
        """Numerically evaluate a Pyomo expression given a solution dict."""
        const, linear, bilinear = extract_bilinear_terms(expr)
        result = const
        for var, coef in linear:
            result += coef * solution.get(self._var_name(var), 0.0)
        for v1, v2, coef in bilinear:
            x = solution.get(self._var_name(v1), 0.0)
            y = solution.get(self._var_name(v2), 0.0)
            result += coef * x * y
        return result


# ---------------------------------------------------------------------------
# Example: bilinear energy scheduling (revenue = price * quantity)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import dimod

    m = pyo.ConcreteModel()

    G = [0, 1, 2]
    T = [0, 1]
    price    = {0: 50.0, 1: 70.0}          # market price $/MWh (variable in real problems)
    cost     = {0: 10,   1: 20,   2: 15}
    pmin     = {0: 20,   1: 10,   2: 30}
    pmax     = {0: 100,  1: 80,   2: 120}
    demand   = {0: 100,  1: 150}

    # Binary on/off
    m.u = pyo.Var(G, T, domain=pyo.Binary)
    # Continuous power output
    m.p = pyo.Var(G, T, domain=pyo.NonNegativeReals,
                bounds=lambda m, g, t: (0, pmax[g]))
    # Continuous price variable (e.g. spot price, bounded)
    m.q = pyo.Var(T, domain=pyo.NonNegativeReals,
                bounds=lambda m, t: (40.0, 80.0))

    # Bilinear objective: maximize revenue (price * output) minus cost
    # revenue = sum_g sum_t q[t] * p[g,t]   <-- bilinear: continuous * continuous
    # cost    = sum_g sum_t cost[g] * p[g,t]
    m.obj = pyo.Objective(
        expr=sum(cost[g] * m.p[g, t] - m.q[t] * m.p[g, t]
                for g in G for t in T),
        sense=pyo.minimize)   # minimize cost - revenue = maximize profit

    # Demand constraints
    m.demand = pyo.Constraint(T, rule=lambda m, t:
        sum(m.p[g, t] for g in G) >= demand[t])

    # Output bounded by on/off
    m.pmin_c = pyo.Constraint(G, T, rule=lambda m, g, t:
        m.p[g, t] >= pmin[g] * m.u[g, t])   # bilinear: continuous * binary
    m.pmax_c = pyo.Constraint(G, T, rule=lambda m, g, t:
        m.p[g, t] <= pmax[g] * m.u[g, t])   # bilinear: continuous * binary

    print("=" * 65)
    print("  Pyomo → QUBO  (bilinear energy scheduling example)")
    print("=" * 65)

    converter = PyomoToQUBO(
        m,
        penalty=2000,
        continuous_bits=5,           # fewer bits = fewer variables; increase for precision
        bilinear_strategy='auto',    # 'expand', 'mccormick', or 'auto'
    )
    bqm = converter.convert()

    print("\n--- Local simulation (SimulatedAnnealing, no D-Wave needed) ---")
    sa = dimod.SimulatedAnnealingSampler()
    ss = sa.sample(bqm, num_reads=1000)
    sol = converter.decode(ss)

    print("\n--- Feasibility Check ---")
    converter.check_feasibility(sol)

    # Uncomment to run on D-Wave:
    # ss = converter.sample(bqm)
    # sol = converter.decode(ss)
    # converter.check_feasibility(sol)
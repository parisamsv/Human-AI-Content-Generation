"""
Core simulation with self-consistency checks on BOTH regime candidates
and scipy-based beta optimization for smooth curves.

Key fix: after computing r* from either the uncovered or covered formula,
verify the result is self-consistent — i.e., the regime assumed in the
derivation actually holds at the computed r*. Discard if not.
"""
import numpy as np
from scipy.optimize import minimize_scalar


# =========================================================================
#  SHARED OPTIMIZER
# =========================================================================

# def _optimize_beta(sim, n_beta, comp):
#     """
#     Two-phase optimizer:
#       1. Grid sweep to find candidate regions
#       2. Scipy refinement for precision
#     """
#     betas = np.linspace(0.0, 1.0, n_beta)
#     utilities = np.zeros(n_beta)

#     for i, b in enumerate(betas):
#         r = sim.optimal_r(b)
#         utilities[i] = sim.utility(b, r)

#     # Collect local maxima + global max as candidates
#     best_idx = np.argmax(utilities)
#     candidates = {betas[best_idx]}
#     for i in range(1, n_beta - 1):
#         if utilities[i] >= utilities[i-1] and utilities[i] >= utilities[i+1]:
#             candidates.add(betas[i])

#     # Scipy refinement around each candidate
#     best = {'beta_h': 0.5, 'r': 0.0, 'utility': -np.inf,
#             'q_h': 0.0, 'q_A': sim.Q}

#     step = 2.0 / n_beta
#     for b_cand in candidates:
#         lo = max(0.0, b_cand - step)
#         hi = min(1.0, b_cand + step)

#         def neg_util(b):
#             return -sim.utility(b, sim.optimal_r(b))

#         try:
#             result = minimize_scalar(neg_util, bounds=(lo, hi),
#                                      method='bounded',
#                                      options={'xatol': 1e-10})
#             b_opt = result.x
#         except Exception:
#             b_opt = b_cand

#         r_opt = sim.optimal_r(b_opt)
#         u_opt = sim.utility(b_opt, r_opt)

#         if u_opt > best['utility']:
#             if comp == 'vc':
#                 qh, qA = sim.qualities(b_opt, r_opt)
#             else:
#                 qh, qA = sim.qualities(r_opt)
#             best = {'beta_h': b_opt, 'r': r_opt, 'utility': u_opt,
#                     'q_h': qh, 'q_A': qA}

#     return best


# =========================================================================
#  VIEW-BASED COMPENSATION
# =========================================================================

class PlatformSimulator_ViewComp:
    def __init__(self, model, alpha, delta, Q, k, u_0):
        self.alpha = alpha
        self.delta = delta
        self.t = 2.5
        self.u_0 = u_0
        self.Q = Q
        self.k = k
        self.model = model

    def mismatch(self, b):
        mH = 1.0 - b * self.delta
        mA = 1.0 - self.delta + b * self.delta
        return mH, mA

    def _coverage_sum(self, b, qh, qA):
        mH, mA = self.mismatch(b)
        dA = max(qA - self.u_0, 0) / (self.t * mA)
        dH = max(qh - self.u_0, 0) / (self.t * mH)
        return dA + dH

    def _is_covered(self, b, qh, qA):
        return self._coverage_sum(b, qh, qA) > 1.0

    def qualities(self, b, r):
        mH, mA = self.mismatch(b)
        S = self.t * (mA + mH)
        qh_unc = r / (self.t * mH)
        qA_unc = self.alpha * qh_unc + self.Q
        if not self._is_covered(b, qh_unc, qA_unc):
            return qh_unc, qA_unc
        qh_cov = r * (1 - self.alpha) / S
        qA_cov = self.alpha * qh_cov + self.Q
        return qh_cov, qA_cov

    def demands(self, b, r):
        mH, mA = self.mismatch(b)
        qh, qA = self.qualities(b, r)
        dA = max(qA - self.u_0, 0) / (self.t * mA)
        dH = max(qh - self.u_0, 0) / (self.t * mH)
        if dA + dH <= 1:
            return max(0.0, min(dH, 1.0)), max(0.0, min(dA, 1.0))
        else:
            x_hat = (qA - qh + self.t * mH) / (self.t * (mA + mH))
            DA = max(0.0, min(x_hat, 1.0))
            return max(0.0, 1.0 - DA), DA

    def _base_utility(self, b, r):
        qh, qA = self.qualities(b, r)
        Dh, DA = self.demands(b, r)
        if self.model == 'view':
            return DA + (1.0 - r) * Dh
        else:
            return qA * DA + (qh - r) * Dh

    def utility(self, b, r):
        return self._base_utility(b, r) - self.k / 2.0 * (b - 0.5) ** 2

    def _vc_creator_utility(self, qh, r, Dh):
        return r * Dh - 0.5 * qh ** 2

    def optimal_r(self, b):
        mH, mA = self.mismatch(b)
        a, t, u0, Q = self.alpha, self.t, self.u_0, self.Q
        if self.model == 'view':
            return self._optimal_r_VR(b, mH, mA, a, t, u0, Q)
        else:
            return self._optimal_r_ER(b, mH, mA, a, t, u0, Q)

    def _vc_boundary_r(self, mH, mA, a, t, u0, Q):
        """Compute r at which coverage_sum = 1 exactly (uncovered VC qualities).
        At this r the uncovered formulas are still valid (market just barely uncovered).
        """
        W = a / (t**2 * mH * mA) + 1.0 / (t**2 * mH**2)
        if W < 1e-12:
            return None
        r_val = (1.0 - (Q - u0) / (t * mA) + u0 / (t * mH)) / W
        if r_val <= 0:
            return None
        qh = r_val / (t * mH)
        if qh < u0:
            return None
        Dh = (qh - u0) / (t * mH)
        if self._vc_creator_utility(qh, r_val, Dh) < -1e-12:
            return None
        return r_val

    def _optimal_r_VR(self, b, mH, mA, a, t, u0, Q):
        # Uncovered candidate only — no boundary needed for VR
        r_unc = 0.5 * (1.0 + u0 * t * mH + a * mH / mA)
        qh_unc = r_unc / (t * mH)
        qA_unc = a * qh_unc + Q
        if not self._is_covered(b, qh_unc, qA_unc):
            Dh_unc = max(qh_unc - u0, 0) / (t * mH)
            if self._vc_creator_utility(qh_unc, r_unc, Dh_unc) >= -1e-12:
                return r_unc
        return 0.0

    def _optimal_r_ER(self, b, mH, mA, a, t, u0, Q):
        sigma = a ** 2 * mH + mA * (1.0 - t * mH)

        # Uncovered candidate
        r_unc = None
        if sigma < -1e-12:
            num = t * mH * (u0 * mA * (1.0 - t * mH)
                            - a * mH * (2.0 * Q - u0))
            r_val = num / (2.0 * sigma)
            if r_val > 0:
                qh_unc = r_val / (t * mH)
                qA_unc = a * qh_unc + Q
                if not self._is_covered(b, qh_unc, qA_unc):
                    Dh_unc = max(qh_unc - u0, 0) / (t * mH)
                    if self._vc_creator_utility(qh_unc, r_val, Dh_unc) >= -1e-12:
                        r_unc = r_val

        # Boundary candidate
        r_bnd = self._vc_boundary_r(mH, mA, a, t, u0, Q)

        candidates = [(0.0, self.utility(b, 0.0))]
        if r_unc is not None:
            candidates.append((r_unc, self.utility(b, r_unc)))
        if r_bnd is not None:
            candidates.append((r_bnd, self.utility(b, r_bnd)))
        return max(candidates, key=lambda x: x[1])[0]

    # def optimize(self, n_beta=1001):
    #     return _optimize_beta(self, n_beta, comp='vc')
    
    def optimize(self, n_beta=1001):
        betas = np.linspace(0.0, 1.0, n_beta)
        best = {'beta_h': 0.5, 'r': 0.0, 'utility': self.utility(0.5, 0.0),
                'q_h': 0.0, 'q_A': self.Q}
        for b in betas:
            r = self.optimal_r(b)
            u = self.utility(b, r)
            if u > best['utility']:
                qh, qA, _ = self.qualities(b, r)
                best = {'beta_h': b, 'r': r, 'utility': u,
                        'q_h': qh, 'q_A': qA}
        return best

    def creator_utility(self, b, r):
        Dh, _ = self.demands(b, r)
        qh, _ = self.qualities(b, r)
        return r * Dh - 0.5 * qh ** 2

    def total_engagement(self, b, r):
        qh, qA = self.qualities(b, r)
        Dh, DA = self.demands(b, r)
        return qA * DA + qh * Dh

    def total_view(self, b, r):
        Dh, DA = self.demands(b, r)
        return DA + Dh


# =========================================================================
#  ENGAGEMENT-BASED COMPENSATION
# =========================================================================

class PlatformSimulator_EngComp:
    def __init__(self, model, alpha, delta, Q, k, u_0):
        self.alpha = alpha
        self.delta = delta
        self.t = 2.5
        self.u_0 = u_0
        self.Q = Q
        self.k = k
        self.model = model

    def mismatch(self, b):
        mH = 1.0 - b * self.delta
        mA = 1.0 - self.delta + b * self.delta
        return mH, mA

    def qualities(self, r):
        qh = r
        qA = self.alpha * qh + self.Q
        return qh, qA

    def _coverage_sum(self, b, qh, qA):
        mH, mA = self.mismatch(b)
        dA = max(qA - self.u_0, 0) / (self.t * mA)
        dH = max(qh - self.u_0, 0) / (self.t * mH)
        return dA + dH

    def _is_covered(self, b, qh, qA):
        return self._coverage_sum(b, qh, qA) > 1.0

    def demands(self, b, r):
        mH, mA = self.mismatch(b)
        qh, qA = self.qualities(r)
        dA = max(qA - self.u_0, 0) / (self.t * mA)
        dH = max(qh - self.u_0, 0) / (self.t * mH)
        if dA + dH <= 1:
            return max(0.0, min(dH, 1.0)), max(0.0, min(dA, 1.0))
        else:
            x_hat = (qA - qh + self.t * mH) / (self.t * (mA + mH))
            DA = max(0.0, min(x_hat, 1.0))
            return max(0.0, 1.0 - DA), DA

    def _base_utility(self, b, r):
        qh, qA = self.qualities(r)
        Dh, DA = self.demands(b, r)
        if self.model == 'view':
            return DA + Dh - r * qh
        else:
            return qA * DA + qh * Dh - r * qh

    def utility(self, b, r):
        return self._base_utility(b, r) - self.k / 2.0 * (b - 0.5) ** 2

    def optimal_r(self, b):
        mH, mA = self.mismatch(b)
        S = self.t * (mA + mH)
        a, t, u0, Q = self.alpha, self.t, self.u_0, self.Q
        if self.model == 'view':
            return self._optimal_r_VR(b, mH, mA, S, a, t, u0, Q)
        else:
            return self._optimal_r_ER(b, mH, mA, S, a, t, u0, Q)

    def _optimal_r_VR(self, b, mH, mA, S, a, t, u0, Q):
        r_unc = 0.5 * (a / (t * mA) + 1.0 / (t * mH))
        qh, qA = self.qualities(r_unc)
        if qh < u0:
            return 0.0
        if not self._is_covered(b, qh, qA):
            return r_unc
        return 0.0

    def _optimal_r_ER(self, b, mH, mA, S, a, t, u0, Q):
        sigma = a ** 2 * mH + mA * (1.0 - t * mH)

        # --- Uncovered candidate (Prop 7) ---
        r_unc = None
        if sigma < -1e-12:
            num = u0 * mA - a * (2.0 * Q - u0) * mH
            r_val = num / (2.0 * sigma)
            if r_val > 0:
                qh, qA = self.qualities(r_val)
                if not self._is_covered(b, qh, qA):
                    r_unc = r_val

        # --- Covered candidate (Prop 8) ---
        r_cov = None
        Psi = (1 - a) ** 2
        if S > Psi:
            num = t * (a * mH + mA) - 2.0 * (1 - a) * Q
            denom = 2.0 * (S - Psi)
            if abs(denom) > 1e-12:
                r_val = num / denom
                if r_val > 0:
                    qh, qA = self.qualities(r_val)
                    if self._is_covered(b, qh, qA):
                        r_cov = r_val

        # --- Boundary candidate: r at which market is exactly covered ---
        # Handles the gap where both formulas fail self-consistency
        r_bnd = None
        W = a / (t * mA) + 1.0 / (t * mH)
        if W > 1e-12:
            r_val = (1.0 - (Q - u0) / (t * mA) + u0 / (t * mH)) / W
            if r_val > 0:
                r_bnd = r_val

        # --- Pick the candidate with highest platform utility ---
        candidates = [(0.0, self.utility(b, 0.0))]
        if r_unc is not None:
            candidates.append((r_unc, self.utility(b, r_unc)))
        if r_cov is not None:
            candidates.append((r_cov, self.utility(b, r_cov)))
        if r_bnd is not None:
            candidates.append((r_bnd, self.utility(b, r_bnd)))

        return max(candidates, key=lambda x: x[1])[0]

    # def optimize(self, n_beta=1001):
    #     return _optimize_beta(self, n_beta, comp='ec')
    
    def optimize(self, n_beta=1001):
        betas = np.linspace(0.0, 1.0, n_beta)
        best = {'beta_h': 0.5, 'r': 0.0, 'utility': self.utility(0.5, 0.0),
                'q_h': 0.0, 'q_A': self.Q}
        for b in betas:
            r = self.optimal_r(b)
            u = self.utility(b, r)
            if u > best['utility']:
                qh, qA, _ = self.qualities(b, r)
                best = {'beta_h': b, 'r': r, 'utility': u,
                        'q_h': qh, 'q_A': qA}
        return best

    def creator_utility(self, b, r):
        qh, _ = self.qualities(r)
        return r * qh - 0.5 * qh ** 2

    def total_engagement(self, b, r):
        qh, qA = self.qualities(r)
        Dh, DA = self.demands(b, r)
        return qA * DA + qh * Dh

    def total_view(self, b, r):
        Dh, DA = self.demands(b, r)
        return DA + Dh
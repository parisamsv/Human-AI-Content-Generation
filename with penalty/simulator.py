"""
Core simulation for view-based and engagement-based models with
discrimination penalty k/2 * (beta_h - 0.5)^2.

IMPORTANT: r is the per-view compensation ratio and must satisfy r ∈ [0, 1].
- View-based:  r* = min(r_hat, 1),  r_hat > 1/2 always so r* > 0 automatic.
- Engagement:  r* = clip(r_hat, 0, 1).  When SOC fails (Sigma >= 0),
               we fall back to comparing u_p at r=0 and r=1.
"""
import numpy as np


class PlatformSimulator:
    def __init__(self, alpha, delta, t, u_0, Q, k, model):
        self.alpha = alpha
        self.delta = delta
        self.t = t
        self.u_0 = u_0
        self.Q = Q
        self.k = k
        self.model = model

    # -- primitives -------------------------------------------------------

    def mismatch(self, b):
        mH = 1.0 - b * self.delta
        mA = 1.0 - self.delta + b * self.delta
        return mH, mA

    def qualities(self, b, r):
        mH, _ = self.mismatch(b)
        qh = r / (self.t * mH)
        qA = self.alpha * qh + self.Q
        return qh, qA

    def demands(self, b, r):
        mH, mA = self.mismatch(b)
        qh, qA = self.qualities(b, r)
        Dh = max(0.0, (qh - self.u_0) / (self.t * mH))
        DA = max(0.0, (qA - self.u_0) / (self.t * mA))
        return Dh, DA

    # -- platform utility (without penalty, for r-optimisation) -----------

    def _base_utility(self, b, r):
        """Platform revenue before penalty."""
        qh, qA = self.qualities(b, r)
        Dh, DA = self.demands(b, r)
        if self.model == 'view':
            return DA + (1.0 - r) * Dh
        else:
            return qA * DA + (qh - r) * Dh

    def utility(self, b, r):
        """Platform payoff including penalty."""
        return self._base_utility(b, r) - self.k / 2.0 * (b - 0.5) ** 2

    # -- analytical r* with proper clipping --------------------------------

    def _sigma(self, b):
        """SOC term for engagement model: alpha^2 * mH + mA * (1 - t*mH)."""
        mH, mA = self.mismatch(b)
        return self.alpha ** 2 * mH + mA * (1.0 - self.t * mH)

    def optimal_r(self, b):
        """
        Closed-form r* clipped to [0, 1].
        View-based:  always concave in r;  r_hat > 0.5.
        Engagement:  concave only when Sigma < 0;  fall back to boundary
                     comparison when Sigma >= 0.
        """
        mH, mA = self.mismatch(b)
        a, t, u0, Q = self.alpha, self.t, self.u_0, self.Q

        if self.model == 'view':
            # Unconstrained maximiser (always > 0.5)
            r_hat = 0.5 * (1.0 + a * mH / mA + t * mH * u0)
            return min(r_hat, 1.0)

        else:  # engagement
            sigma = self._sigma(b)

            if sigma >= 0:
                # SOC fails: payoff is convex (or linear) in r.
                # Optimum is at a boundary of [0, 1].
                u0_val = self._base_utility(b, 0.0)
                u1_val = self._base_utility(b, 1.0)
                return 1.0 if u1_val >= u0_val else 0.0

            # sigma < 0: concave, FOC gives maximum
            num = t * mH * (u0 * mA * (1.0 - t * mH)
                            - a * mH * (2.0 * Q - u0))
            r_hat = num / (2.0 * sigma)
            return min(r_hat, 1.0)

    # -- optimise over beta -----------------------------------------------

    def optimize(self, n_beta=1001):
        betas = np.linspace(0.0, 1.0, n_beta)
        best = {'beta_h': 0.5, 'r': 0.0, 'utility': -np.inf,
                'q_h': 0.0, 'q_A': 0.0}
        for b in betas:
            r = self.optimal_r(b)
            u = self.utility(b, r)
            if u > best['utility']:
                qh, qA = self.qualities(b, r)
                best = {'beta_h': b, 'r': r, 'utility': u,
                        'q_h': qh, 'q_A': qA}
        return best

    def optimize_general(self, n_beta=201, n_r=201):
        """
        Brute-force grid over (beta, r) ∈ [0,1]^2.
        No analytical assumptions.  Use for robustness checks.
        """
        betas = np.linspace(0.0, 1.0, n_beta)
        rs = np.linspace(0.0, 1.0, n_r)
        best = {'beta_h': 0.5, 'r': 0.0, 'utility': -np.inf,
                'q_h': 0.0, 'q_A': 0.0}
        for b in betas:
            for r in rs:
                u = self.utility(b, r)
                if u > best['utility']:
                    qh, qA = self.qualities(b, r)
                    best = {'beta_h': b, 'r': r, 'utility': u,
                            'q_h': qh, 'q_A': qA}
        return best

    # -- welfare -----------------------------------------------------------

    def consumer_surplus(self, b, r):
        mH, mA = self.mismatch(b)
        qh, qA = self.qualities(b, r)
        cs_A = max(0.0, (qA - self.u_0)) ** 2 / (2.0 * self.t * mA)
        cs_H = max(0.0, (qh - self.u_0)) ** 2 / (2.0 * self.t * mH)
        return cs_A + cs_H

    def creator_welfare(self, b, r):
        Dh, _ = self.demands(b, r)
        qh, _ = self.qualities(b, r)
        return r * Dh - 0.5 * qh ** 2

    def total_quality(self, b, r):
        qh, qA = self.qualities(b, r)
        Dh, DA = self.demands(b, r)
        return qA * DA + qh * Dh

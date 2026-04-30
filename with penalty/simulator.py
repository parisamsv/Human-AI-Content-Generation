"""
Core simulation for view-based and engagement-based models with
discrimination penalty k/2 * (beta_h - 0.5)^2.

Four models:
  ViewComp + ViewRev   (VC+VR)
  ViewComp + EngRev    (VC+ER)
  EngComp  + ViewRev   (EC+VR)
  EngComp  + EngRev    (EC+ER)

Each model has closed-form r* for uncovered and covered regimes.
We compute the uncovered candidate, check self-consistency, and
fall back to the covered candidate if needed.
"""
import numpy as np


# =========================================================================
#  VIEW-BASED COMPENSATION
# =========================================================================

class PlatformSimulator_ViewComp:
    def __init__(self, alpha, delta, Q, k, model):
        self.alpha = alpha
        self.delta = delta
        self.t = 3.5
        self.u_0 = 0
        self.Q = Q
        self.k = k
        self.model = model  # 'view' (VR) or 'engagement' (ER)

    # -- primitives -------------------------------------------------------

    def mismatch(self, b):
        mH = 1.0 - b * self.delta
        mA = 1.0 - self.delta + b * self.delta
        return mH, mA

    def _is_covered(self, b, qh, qA):
        """Check whether given qualities produce a covered market."""
        mH, mA = self.mismatch(b)
        dA = max(qA - self.u_0, 0) / (self.t * mA)
        dH = max(qh - self.u_0, 0) / (self.t * mH)
        return dA + dH > 1.0

    def qualities(self, b, r):
        """Compute equilibrium qualities, auto-detecting regime."""
        mH, mA = self.mismatch(b)
        S = self.t * (mA + mH)

        # Try uncovered first
        qh_unc = r / (self.t * mH)
        qA_unc = self.alpha * qh_unc + self.Q

        if not self._is_covered(b, qh_unc, qA_unc):
            return qh_unc, qA_unc, True  # uncovered

        # Covered
        qh_cov = r * (1 - self.alpha) / S
        qA_cov = self.alpha * qh_cov + self.Q
        return qh_cov, qA_cov, False  # covered

    def demands(self, b, r):
        """Compute equilibrium demands, consistent with quality regime."""
        mH, mA = self.mismatch(b)
        qh, qA, uncovered = self.qualities(b, r)

        if uncovered:
            DA = max(0.0, (qA - self.u_0) / (self.t * mA))
            Dh = max(0.0, (qh - self.u_0) / (self.t * mH))
        else:
            x_hat = (qA - qh + self.t * mH) / (self.t * (mA + mH))
            DA = max(0.0, min(x_hat, 1.0))
            Dh = max(0.0, min(1.0 - x_hat, 1.0))

        return Dh, DA, uncovered

    # -- platform utility -------------------------------------------------

    def _base_utility(self, b, r):
        """Platform revenue before penalty."""
        qh, qA, _ = self.qualities(b, r)
        Dh, DA, _ = self.demands(b, r)
        if self.model == 'view':
            return DA + (1.0 - r) * Dh
        else:
            return qA * DA + (qh - r) * Dh

    def utility(self, b, r):
        """Platform payoff including penalty."""
        return self._base_utility(b, r) - self.k / 2.0 * (b - 0.5) ** 2

    # -- analytical r* ----------------------------------------------------

    def optimal_r(self, b):
        mH, mA = self.mismatch(b)
        S = self.t * (mA + mH)
        a, t, u0, Q = self.alpha, self.t, self.u_0, self.Q

        if self.model == 'view':
            return self._optimal_r_VR(b, mH, mA, S, a, t, u0, Q)
        else:
            return self._optimal_r_ER(b, mH, mA, S, a, t, u0, Q)

    def _optimal_r_VR(self, b, mH, mA, S, a, t, u0, Q):
        """VC + VR: Propositions 1 (uncovered) and 2 (covered)."""
        # Uncovered candidate
        r_unc = 0.5 * (1.0 + a * mH / mA + t * mH * u0)
        qh_unc = r_unc / (t * mH)

        if qh_unc < u0:
            r_unc = 0.0

        # Check regime with uncovered r*
        if r_unc > 0:
            qA_unc = a * qh_unc + Q
            if not self._is_covered(b, qh_unc, qA_unc):
                return r_unc

        # Covered candidate (Prop 2): requires Q > t*mA
        if Q <= t * mA:
            return 0.0
        r_cov = S * (Q - t * mA) / (2.0 * (1 - a) ** 2)
        return max(r_cov, 0.0)

    def _optimal_r_ER(self, b, mH, mA, S, a, t, u0, Q):
        """VC + ER: Propositions 3 (uncovered) and 4 (covered)."""
        sigma = a ** 2 * mH + mA * (1.0 - t * mH)

        # Uncovered candidate (Prop 3): need sigma < 0 for SOC
        r_unc = None
        if sigma < -1e-12:
            num = t * mH * (u0 * mA * (1.0 - t * mH)
                            - a * mH * (2.0 * Q - u0))
            r_unc = num / (2.0 * sigma)
            if r_unc < 0:
                r_unc = 0.0

        # Check regime
        if r_unc is not None and r_unc > 0:
            qh_unc = r_unc / (t * mH)
            qA_unc = a * qh_unc + Q
            if not self._is_covered(b, qh_unc, qA_unc):
                return r_unc

        # Covered candidate (Prop 4)
        Phi = (1 - a) / S
        Psi = (1 - a) ** 2 / S
        if Psi >= 1.0:
            return 0.0  # SOC fails
        A = Q + t * mH
        B = t * mA - Q
        num = a * Phi * A + (Phi - 1) * B - Q * Psi
        denom = 2.0 * Psi * (1 - Psi)
        if abs(denom) < 1e-12:
            return 0.0
        r_cov = num / denom
        return max(r_cov, 0.0)

    # -- optimise over beta -----------------------------------------------

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

    # -- metrics -----------------------------------------------------------

    def creator_utility(self, b, r):
        Dh, _, _ = self.demands(b, r)
        qh, _, _ = self.qualities(b, r)
        return r * Dh - 0.5 * qh ** 2

    def total_engagement(self, b, r):
        qh, qA, _ = self.qualities(b, r)
        Dh, DA, _ = self.demands(b, r)
        return qA * DA + qh * Dh

    def total_view(self, b, r):
        Dh, DA, _ = self.demands(b, r)
        return DA + Dh


# =========================================================================
#  ENGAGEMENT-BASED COMPENSATION
# =========================================================================

class PlatformSimulator_EngComp:
    def __init__(self, alpha, delta, Q, k, model):
        self.alpha = alpha
        self.delta = delta
        self.t = 3.5
        self.u_0 = 0
        self.Q = Q
        self.k = k
        self.model = model  # 'view' (VR) or 'engagement' (ER)

    # -- primitives -------------------------------------------------------

    def mismatch(self, b):
        mH = 1.0 - b * self.delta
        mA = 1.0 - self.delta + b * self.delta
        return mH, mA

    def qualities(self, r):
        """EC: q_h = r always (demand-independent)."""
        qh = r
        qA = self.alpha * qh + self.Q
        return qh, qA

    def _is_covered(self, b, qh, qA):
        """Check whether given qualities produce a covered market."""
        mH, mA = self.mismatch(b)
        dA = max(qA - self.u_0, 0) / (self.t * mA)
        dH = max(qh - self.u_0, 0) / (self.t * mH)
        return dA + dH > 1.0

    def demands(self, b, r):
        """Compute demands using q_h = r (EC is regime-independent)."""
        mH, mA = self.mismatch(b)
        qh, qA = self.qualities(r)  # FIX: was self.qualities(b, r)

        dA = max(qA - self.u_0, 0) / (self.t * mA)
        dH = max(qh - self.u_0, 0) / (self.t * mH)

        if dA + dH <= 1:
            DA = max(0.0, dA)
            Dh = max(0.0, dH)
            uncovered = True
        else:
            x_hat = (qA - qh + self.t * mH) / (self.t * (mA + mH))
            DA = max(0.0, min(x_hat, 1.0))
            Dh = max(0.0, min(1.0 - x_hat, 1.0))
            uncovered = False

        return Dh, DA, uncovered

    # -- platform utility -------------------------------------------------

    def _base_utility(self, b, r):
        """Platform revenue before penalty."""
        qh, qA = self.qualities(r)
        Dh, DA, _ = self.demands(b, r)
        if self.model == 'view':
            return DA + Dh - r * qh
        else:
            return qA * DA + qh * Dh - r * qh

    def utility(self, b, r):
        """Platform payoff including penalty."""
        return self._base_utility(b, r) - self.k / 2.0 * (b - 0.5) ** 2

    # -- analytical r* ----------------------------------------------------

    def optimal_r(self, b):
        mH, mA = self.mismatch(b)
        S = self.t * (mA + mH)
        a, t, u0, Q = self.alpha, self.t, self.u_0, self.Q

        if self.model == 'view':
            return self._optimal_r_VR(b, mH, mA, S, a, t, u0, Q)
        else:
            return self._optimal_r_ER(b, mH, mA, S, a, t, u0, Q)

    def _optimal_r_VR(self, b, mH, mA, S, a, t, u0, Q):
        """EC + VR: Propositions 5 (uncovered) and 6 (covered, r*=0)."""
        # Uncovered candidate
        r_unc = 0.5 * (a / (t * mA) + 1.0 / (t * mH))
        qh, qA = self.qualities(r_unc)

        if qh < u0:
            return 0.0

        if not self._is_covered(b, qh, qA):
            return r_unc

        # Covered: r* = 0 (Prop 6)
        return 0.0

    def _optimal_r_ER(self, b, mH, mA, S, a, t, u0, Q):
        """EC + ER: Propositions 7 (uncovered) and 8 (covered)."""
        # FIX: use paper's denominator directly (was sign-flipped)
        sigma = a ** 2 * mH + mA * (1.0 - t * mH)

        # Uncovered candidate (Prop 7): need sigma < 0 for SOC
        r_unc = None
        if sigma < -1e-12:
            num = u0 * mA - a * (2.0 * Q - u0) * mH
            r_unc = num / (2.0 * sigma)  # neg/neg = positive
            if r_unc < 0:
                r_unc = 0.0

        # Check regime
        if r_unc is not None and r_unc > 0:
            qh, qA = self.qualities(r_unc)
            if not self._is_covered(b, qh, qA):
                return r_unc

        # Covered candidate (Prop 8)
        Psi = (1 - a) ** 2
        if S <= Psi:
            return 0.0  # SOC fails
        num = t * (a * mH + mA) - 2.0 * (1 - a) * Q
        denom = 2.0 * (S - Psi)
        if abs(denom) < 1e-12:
            return 0.0
        r_cov = num / denom
        return max(r_cov, 0.0)

    # -- optimise over beta -----------------------------------------------

    def optimize(self, n_beta=1001):
        betas = np.linspace(0.0, 1.0, n_beta)
        best = {'beta_h': 0.5, 'r': 0.0, 'utility': self.utility(0.5, 0.0),
                'q_h': 0.0, 'q_A': self.Q}
        for b in betas:
            r = self.optimal_r(b)
            u = self.utility(b, r)
            if u > best['utility']:
                qh, qA = self.qualities(r)
                best = {'beta_h': b, 'r': r, 'utility': u,
                        'q_h': qh, 'q_A': qA}
        return best

    # -- metrics -----------------------------------------------------------

    def creator_utility(self, b, r):
        qh, _ = self.qualities(r)
        return r * qh - 0.5 * qh ** 2

    def total_engagement(self, b, r):
        qh, qA = self.qualities(r)
        Dh, DA, _ = self.demands(b, r)
        return qA * DA + qh * Dh

    def total_view(self, b, r):
        Dh, DA, _ = self.demands(b, r)
        return DA + Dh
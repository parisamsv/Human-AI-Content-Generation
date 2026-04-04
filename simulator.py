"""
Core simulation logic for comparing platform revenue settings.
"""

import numpy as np
from scipy.optimize import minimize_scalar

class PlatformRevenueSimulator:
    """
    Simulates two platform revenue settings for human-AI content generation.
    """
    
    def __init__(self, t=1.0, alpha=0.6, Q=0.0, u_0=0.5, delta=0.4):
        """
        Initialize platform parameters.
        """
        self.t = t
        self.alpha = alpha
        self.Q = Q
        self.u_0 = u_0
        self.delta = delta
    
    def mismatch_costs(self, beta_h):
        """Calculate mismatch costs given promotion weight."""
        m_H = 1 - beta_h * self.delta
        m_A = 1 - self.delta + beta_h * self.delta
        return m_H, m_A
    
    def creator_qualities(self, beta_h, r):
        """
        Stage 2: Calculate optimal creator qualities.
        """
        m_H, _ = self.mismatch_costs(beta_h)
        q_h = r / (self.t * m_H)
        q_A = (self.alpha * r) / (self.t * m_H) + self.Q
        return q_h, q_A
    
    def demands(self, beta_h, r):
        """
        Calculate demands with boundary conditions.
        """
        m_H, m_A = self.mismatch_costs(beta_h)
        q_h, q_A = self.creator_qualities(beta_h, r)
        
        D_h = max(0, (q_h - self.u_0) / (self.t * m_H))
        D_A = max(0, (q_A - self.u_0) / (self.t * m_A))
        
        return D_h, D_A
    
    def platform_utility_view(self, beta_h, r):
        """
        Setting 1: View-based revenue (total volume).
        """
        D_h, D_A = self.demands(beta_h, r)
        return D_A + (1 - r) * D_h
    
    def platform_utility_engagement(self, beta_h, r):
        """
        Setting 2: Engagement-based revenue.
        """
        q_h, q_A = self.creator_qualities(beta_h, r)
        D_h, D_A = self.demands(beta_h, r)
        
        engagement = q_A * D_A + (q_h - r) * D_h
        return engagement
    
    def optimal_r_engagement(self, beta_h):
        """
        Setting 2: Analytically derived optimal compensation.
        """
        m_H, m_A = self.mismatch_costs(beta_h)
        
        numerator = (self.t * m_H * 
                    (self.u_0 * m_A * (1 - self.t * m_H) - 
                     self.alpha * m_H * (2 * self.Q - self.u_0)))
        
        denominator = 2 * (self.alpha**2 * self.t * m_H + 
                          m_A * (1 - self.t * m_H))
        
        if denominator == 0:
            return 0

        r_opt = numerator / denominator
        return max(0, r_opt)

    def optimal_r_view(self, beta_h):
        """
        Setting 1: Analytically derived optimal compensation.
        """
        m_H, m_A = self.mismatch_costs(beta_h)
        
        return 0.5 * ( 1 + (self.alpha * m_H)/m_A + self.t * m_H * self.u_0 )
    
    def optimize_view_based(self):
        """
        Setting 1: Optimize view-based revenue.
        """
        
        best_utility = -np.inf
        best_beta_h = 0
        best_r = 0
        
        beta_h_values = np.linspace(0, 1, 51)
        
        for beta_h in beta_h_values:
            r_candidate = self.optimal_r_view(beta_h)
            utility_candidate = self.platform_utility_view(beta_h, r_candidate)
            
            if utility_candidate > best_utility:
                best_utility = utility_candidate
                best_beta_h = beta_h
                best_r = r_candidate
        
        q_h, q_A = self.creator_qualities(best_beta_h, best_r)
        
        return {
            'beta_h': best_beta_h,
            'r': best_r,
            'utility': best_utility,
            'q_h': q_h,
            'q_A': q_A
        }
    
    def optimize_engagement_based(self):
        """
        Setting 2: Optimize engagement-based revenue.
        """
        best_utility = -np.inf
        best_beta_h = 0
        best_r = 0

        beta_h_values = np.linspace(0, 1, 101)
        
        for beta_h in beta_h_values:
            r_candidate = self.optimal_r_engagement(beta_h)
            utility_candidate = self.platform_utility_engagement(beta_h, r_candidate)
            
            if utility_candidate > best_utility:
                best_utility = utility_candidate
                best_beta_h = beta_h
                best_r = r_candidate
        
        q_h, q_A = self.creator_qualities(best_beta_h, best_r)
        
        return {
            'beta_h': best_beta_h,
            'r': best_r,
            'utility': best_utility,
            'q_h': q_h,
            'q_A': q_A
        }

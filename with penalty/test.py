"""
Run this to verify the line-209 fix is correctly applied.
It checks: (1) the exact code on line 209, (2) numerical output, 
(3) cached .pyc files that might override your edit.
"""
import os, sys, importlib

# ---- Step 1: Check for stale .pyc cache ----
pyc_dir = os.path.join(os.path.dirname(__file__), '__pycache__')
if os.path.isdir(pyc_dir):
    pyc_files = [f for f in os.listdir(pyc_dir) if 'simulator' in f]
    if pyc_files:
        print("WARNING: Found cached .pyc files that may override your edit:")
        for f in pyc_files:
            full = os.path.join(pyc_dir, f)
            print(f"  {full}")
        print("  --> Deleting them now...")
        for f in pyc_files:
            os.remove(os.path.join(pyc_dir, f))
        print("  --> Deleted. Re-importing simulator.\n")

# Force reimport
if 'simulator' in sys.modules:
    del sys.modules['simulator']

# ---- Step 2: Inspect the actual source code on line 209 ----
with open('simulator.py', 'r') as f:
    lines = f.readlines()

print("=== Line 209 of simulator.py ===")
print(f"  {lines[208].rstrip()}")
print()

# Check for the bug pattern
line209 = lines[208]
if 'self.alpha / t * mA' in line209 or 'alpha / t * mA' in line209:
    print("BUG STILL PRESENT: 'alpha / t * mA' evaluates as (alpha/t)*mA")
    print("FIX: Change to 'self.alpha / (t * mA)'")
    print()
elif 'self.alpha / (t * mA)' in line209 and '1.0 / (t * mH)' in line209:
    print("FIX LOOKS CORRECT (parenthesised form)")
    print()
elif 'self.alpha / t / mA' in line209:
    print("FIX LOOKS CORRECT (chained division form)")
    print()
else:
    print("UNRECOGNIZED FORM — check manually. The correct formula is:")
    print("  r_hat = 0.5 * (self.alpha / (t * mA) + 1.0 / (t * mH))")
    print()

# ---- Step 3: Numerical verification ----
from simulator import PlatformSimulator_ViewComp, PlatformSimulator_EngComp
import numpy as np

alpha, delta, Q, k = 0.5, 0.3, 0.3, 1.0

print("=== Numerical test (view revenue, default params) ===")

sim_vc = PlatformSimulator_ViewComp(alpha, delta, Q, k, 'view')
sim_ec = PlatformSimulator_EngComp(alpha, delta, Q, k, 'view')

# Test at beta_h = 0.5
b = 0.5
r_vc = sim_vc.optimal_r(b)
r_ec = sim_ec.optimal_r(b)

mH, mA = sim_vc.mismatch(b)
t = sim_vc.t

# What the values SHOULD be:
r_vc_expected = 0.5 * (1.0 + alpha * mH / mA)
r_ec_expected = 0.5 * (alpha / (t * mA) + 1.0 / (t * mH))

# Check effort equivalence
qh_vc = r_vc / (t * mH)
qh_ec = r_ec  # since qh = r in EngComp

print(f"  ViewComp r*: {r_vc:.6f}  (expected {r_vc_expected:.6f})  {'OK' if abs(r_vc - r_vc_expected) < 1e-10 else 'WRONG'}")
print(f"  EngComp  r*: {r_ec:.6f}  (expected {r_ec_expected:.6f})  {'OK' if abs(r_ec - r_ec_expected) < 1e-10 else 'WRONG'}")
print()
print(f"  ViewComp effort (r/t/mH): {qh_vc:.6f}")
print(f"  EngComp  effort (r):      {qh_ec:.6f}")
print(f"  Effort difference:        {abs(qh_vc - qh_ec):.2e}")
print()

if abs(qh_vc - qh_ec) < 1e-10:
    print("PASS: Efforts match. Equivalence holds.")
else:
    print("FAIL: Efforts differ. The fix is NOT correctly applied.")
    print(f"  EngComp computes r* = {r_ec:.6f}")
    print(f"  But should be    r* = {r_ec_expected:.6f}")
    print(f"  If your r* is    r* = {0.5*(alpha/t*mA + 1/t*mH):.6f} then the bug is still there.")

# ---- Step 4: Full optimize comparison ----
print("\n=== Full optimization comparison ===")
res_vc = sim_vc.optimize()
res_ec = sim_ec.optimize()

for key in ['beta_h', 'q_h', 'q_A', 'utility']:
    diff = abs(res_vc[key] - res_ec[key])
    status = 'MATCH' if diff < 1e-6 else 'MISMATCH'
    print(f"  {key:10s}: VC={res_vc[key]:.6f}  EC={res_ec[key]:.6f}  {status}")

cu_vc = sim_vc.creator_utility(res_vc['beta_h'], res_vc['r'])
cu_ec = sim_ec.creator_utility(res_ec['beta_h'], res_ec['r'])
diff = abs(cu_vc - cu_ec)
status = 'MATCH' if diff < 1e-6 else 'MISMATCH'
print(f"  {'creator_u':10s}: VC={cu_vc:.6f}  EC={cu_ec:.6f}  {status}")
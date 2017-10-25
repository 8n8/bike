import sympy as s  # type: ignore

B1 = s.Symbol('B1')
B2 = s.Symbol('B2')
M1 = s.Symbol('M1')
M2 = s.Symbol('M2')
P1 = s.Symbol('P1')
P2 = s.Symbol('P2')
S1 = s.Symbol('S1')
S2 = s.Symbol('S2')
k1 = s.Symbol('k1')
k2 = s.Symbol('k2')
n1 = s.Symbol('n1')
n2 = s.Symbol('n2')
r = s.Symbol('r')
t1 = s.Symbol('t1')
t2 = s.Symbol('t2')

equations = [
    B1 - S1 - k1,
    B2 - S2 - k2,
    n1 + P1 - M1 - B1 - t1,
    n2 + P2 - M2 - B2 - t2,
    S1 * k1 + S2 * k2,
    P1 * M1 + P2 * M2,
    P1 * B1 + P2 * B2,
    P1**2 + P2**2 - r**2]

unknowns = [B1, B2, S1, S2, M1, M2, P1, P2]

print(s.solve(equations, unknowns))

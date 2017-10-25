import sympy as s  # type: ignore

A1 = s.Symbol('A1')
A2 = s.Symbol('A2')
B1 = s.Symbol('B1')
B2 = s.Symbol('B2')
C1 = s.Symbol('C1')
C2 = s.Symbol('C2')
D1 = s.Symbol('D1')
D2 = s.Symbol('D2')
F1 = s.Symbol('F1')
F2 = s.Symbol('F2')
G1 = s.Symbol('G1')
G2 = s.Symbol('G2')
M1 = s.Symbol('M1')
M2 = s.Symbol('M2')
N1 = s.Symbol('N1')
N2 = s.Symbol('N2')
P1 = s.Symbol('P1')
P2 = s.Symbol('P2')
Q1 = s.Symbol('Q1')
Q2 = s.Symbol('Q2')
t1 = s.Symbol('t1')
t2 = s.Symbol('t2')
k1 = s.Symbol('k1')
k2 = s.Symbol('k2')
n1 = s.Symbol('n1')
n2 = s.Symbol('n2')
r = s.Symbol('r')
cos_half_theta = s.Symbol('cos_half_theta')

equations = [
    C1 + F1 - B1,
    C2 + F2 - B2,
    k1 + G1 - C1,
    k2 + G2 - C2,
    F1 + M1 - P1 + Q1 - N1,
    F2 + M2 - P2 + Q2 - N2,
    t1 + B1 + M1 - P1 - n1,
    t2 + B2 + M2 - P2 - n2,
    k1 * G1 + k2 * G2,
    k1 * F1 + k2 * F2,
    P1 * M1 + P2 * M2,
    P1 * B1 + P2 * B2,
    Q1 * N1 + A2 * N2,
    Q1 * C1 + Q2 * C2,
    P1**2 + P2**2 - r**2,
    Q1**2 + Q2**2 - r**2]

unknowns = [B1, B2, C1, C2, F1, F2, G1, G2, M1, M2, N1, N2, P1, P2, Q1, Q2]

print(s.solve(equations, unknowns))

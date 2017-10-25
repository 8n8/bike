import sympy as s  # type: ignore

C1 = s.Symbol('C1')
C2 = s.Symbol('C2')
G1 = s.Symbol('G1')
G2 = s.Symbol('G2')
N1 = s.Symbol('N1')
N2 = s.Symbol('N2')
Q1 = s.Symbol('Q1')
Q2 = s.Symbol('Q2')
t1 = s.Symbol('t1')
t2 = s.Symbol('t2')
k1 = s.Symbol('k1')
k2 = s.Symbol('k2')
n1 = s.Symbol('n1')
n2 = s.Symbol('n2')
r = s.Symbol('r')

equations = [
    C1 - G1 - k1,
    C2 - G2 - k2,
    n1 + Q1 - N1 - C1 - t1,
    n2 + Q2 - N2 - C2 - t2,
    G1 * k1 + G2 * k2,
    Q1 * N1 + Q2 * N2,
    Q1 * C1 + Q2 * C2,
    Q1**2 + Q2**2 - r**2]

unknowns = [C1, C2, G1, G2, N1, N2, Q1, Q2]

print(s.solve(equations, unknowns))

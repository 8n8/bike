import sympy as s  # type: ignore

a1 = s.Symbol('a1')
a2 = s.Symbol('a2')
R1 = s.Symbol('R1')
R2 = s.Symbol('R2')
B1 = s.Symbol('B1')
B2 = s.Symbol('B2')
C1 = s.Symbol('C1')
C2 = s.Symbol('C2')
b1 = s.Symbol('b1')
b2 = s.Symbol('b2')
k1 = s.Symbol('k1')
k2 = s.Symbol('k2')
D1 = s.Symbol('D1')
D2 = s.Symbol('D2')
A1 = s.Symbol('A1')
A2 = s.Symbol('A2')
r = s.Symbol('r')

equations = [
    a1 + R1 + B1 - C1 - b1,
    a2 + R2 + B2 - C2 - b2,
    k1 + D1 - C1,
    k2 + D2 - C2,
    A1 - C1 - b1,
    A2 - C2 - b2,
    B1 * R1 + B2 * R2,
    k1 * D1 + k2 * D2,
    R1**2 + R2**2 - r**2,
    C1 * R1 + C2 * R2]

print(s.solve(equations, [R1, R2, B1, B2, C1, C2, D1, D2, A1, A2]))

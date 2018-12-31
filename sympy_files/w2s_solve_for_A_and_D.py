import sympy as s  # type: ignore

A1 = s.Symbol('A1')
A2 = s.Symbol('A2')
D1 = s.Symbol('D1')
D2 = s.Symbol('D2')
k1 = s.Symbol('k1')
k2 = s.Symbol('k2')
cos_half_theta = s.Symbol('costheta')

equations = [
    A1 + D1 - 2 * k1,
    A2 + D2 - 2 * k2,
    cos_half_theta - ((k1**2 + k2**2) / (A1**2 + A2**2))**0.5,
    cos_half_theta - ((k1**2 + k2**2) / (D1**2 + D2**2))**0.5]

unknowns = [A1, A2, D1, D2]

print(s.solve(equations, unknowns))

import sympy as s  # type: ignore

A1 = s.Symbol('A1')
A2 = s.Symbol('A2')
B1 = s.Symbol('B1')
B2 = s.Symbol('B2')
R1 = s.Symbol('R1')
R2 = s.Symbol('R2')
a1 = s.Symbol('a1')
a2 = s.Symbol('a2')
k1 = s.Symbol('k1')
k2 = s.Symbol('k2')
b1 = s.Symbol('b1')
b2 = s.Symbol('b2')
r = s.Symbol('r')

equations = [
    a1+R1+B1-A1-k1-b1,
    a2+R2+B2-A2-k2-b2,
    R1**2+R2**2-r**2,
    R1*B1+R2*B2,
    k1*A1+k2*A2,
    R1*(k1+A1)+R2*(k2+A2)]

print(s.solve(equations, [A1, A2, B1, B2, R1, R2]))

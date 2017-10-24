import world2sensor as w

a1 = 13
a2 = 9.6
b1 = 2.3
b2 = 2.1
k1 = 1.6
k2 = 7.7
r = 5.2

# print(w._calculate_A1(a1, a2, b1, b2, k1, k2, r))
# print(w._calculate_A2(a1, a2, b1, b2, k1, k2, r))
print(w._calculate_B_new(a1, a2, b1, b2, k1, k2, r))

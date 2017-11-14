import math as m


def main(double k1, double k2, double n1, double n2, double r, double t1, double t2, double theta):
    """
    It works out the vectors needed for creating the camera images, using
    the configuration of the camera and the position of the obstacle.

    The vectors are drawn in the diagram in w2s_vector_diagram.pdf. The
    z-axis is perpendicular to the page and positive when pointing towards
    the reader.  The unit vector pointing in the positive z-direction is
    k.  The required vectors are A, B, C, D, P and Q.  All the variables
    with lower-case names are known.

    The equations for finding A and D are:

                A + D - 2K = 0      v1

        cos(ϴ/2) - |K|/|A| = 0      s1
        cos(ϴ/2) - |K|/|D| = 0      s2

    From the diagram, the unknowns needed for finding C are:

        C, G, N, Q

    The equations are:

                C - G - K = 0
        n + Q - N - C - t = 0

                    G . K = 0
                    Q . N = 0
                    Q . C = 0
                      |Q| = r

    Let G + F = S, then the unknowns needed for finding B are:

        B, S, M, P

    The equations are:

                B - S - K = 0
        n + P - M - B - t = 0

                    S . K = 0
                    B . P = 0
                    M . P = 0
                      |P| = r

    These equations were solved using sympy, a symbolic numeric algebra
    library for Python.  The files containing the code are named
    'w2s_solve_for_*.py' where * is the vector name.  The corresponding
    solution files end in 'txt'.
    """
    cdef double t12, t22, n12, n22, k12, k13, k22, k23, r2, 
    cdef double cos_half_theta, sqrt, BCdenominator
    t12 = t1**2
    t22 = t2**2
    n12 = n1**2
    n22 = n2**2
    k12 = k1**2
    k13 = k1**3
    k22 = k2**2
    k23 = k2**3
    r2 = r**2
    cos_half_theta = m.cos(theta/2)
    sqrt = m.sqrt(n12 - 2*n1*t1 + n22 - 2*n2*t2 - r2 + t12 + t22)
    BCdenominator = (
        k12*n12 - 2*k12*n1*t1 - k12*r2 + k12*t12
        + 2*k1*k2*n1*n2 - 2*k1*k2*n1*t2 - 2*k1*k2*n2*t1
        + 2*k1*k2*t1*t2 + k22*n22 - 2*k22*n2*t2
        - k22*r2 + k22*t22)
    return {
        'P': {
            'x': (r*(-n1*r - n2*sqrt + r*t1 + t2*sqrt)
                  / (n12 - 2*n1*t1 + n22 - 2*n2*t2 + t12 + t22)),
            'y': (-r*(r*(n2 - t2) - (n1 - t1)*sqrt)
                  / (n12 - 2*n1*t1 + n22 - 2*n2*t2 + t12 + t22))},
        'Q': {
            'x': (r*(-n1*r + n2*sqrt + r*t1 - t2*sqrt)
                  / (n12 - 2*n1*t1 + n22 - 2*n2*t2 + t12 + t22)),
            'y': (-r*(r*(n2 - t2) + (n1 - t1) * sqrt)
                  / (n12 - 2*n1*t1 + n22 - 2*n2*t2 + t12 + t22))},
        'A': {
            'x': k1 - k2*m.sqrt(-cos_half_theta**2 + 1.0)/cos_half_theta,
            'y': k2 + k1*m.sqrt(-cos_half_theta**2 + 1.0)/cos_half_theta},
        'B': {
            'x': ((k13*n12 - 2*k13*n1*t1 - k13*r2 + k13*t12
                   + k12*k2*n1*n2 - k12*k2*n1*t2 - k12*k2*n2*t1
                   - k12*k2*r
                   * sqrt
                   + k12*k2*t1*t2 + k1*k22*n12 - 2*k1*k22*n1*t1
                   - k1*k22*r2 + k1*k22*t12 + k23*n1*n2
                   - k23*n1*t2 - k23*n2*t1 - k23*r
                   * sqrt
                   + k23*t1*t2)
                  / BCdenominator),
            'y': ((k13*n1*n2 - k13*n1*t2 - k13*n2*t1 + k13*r
                   * sqrt
                   + k13*t1*t2 + k12*k2*n22 - 2*k12*k2*n2*t2
                   - k12*k2*r2 + k12*k2*t22 + k1*k22*n1*n2
                   - k1*k22*n1*t2 - k1*k22*n2*t1 + k1*k22*r
                   * sqrt
                   + k1*k22*t1*t2 + k23*n22 - 2*k23*n2*t2
                   - k23*r2 + k23*t22)
                  / BCdenominator)},
        'C': {
            'x': ((k13*n12 - 2*k13*n1*t1 - k13*r2 + k13*t12
                   + k12*k2*n1*n2 - k12*k2*n1*t2 - k12*k2*n2*t1
                   + k12*k2*r*sqrt
                   + k12*k2*t1*t2 + k1*k22*n12 - 2*k1*k22*n1*t1
                   - k1*k22*r2 + k1*k22*t12 + k23*n1*n2
                   - k23*n1*t2 - k23*n2*t1 + k23*r*sqrt
                   + k23*t1*t2)
                  / BCdenominator),
            'y': ((k13*n1*n2 - k13*n1*t2 - k13*n2*t1 - k13*r
                   * sqrt
                   + k13*t1*t2 + k12*k2*n22 - 2*k12*k2*n2*t2
                   - k12*k2*r2 + k12*k2*t22 + k1*k22*n1*n2
                   - k1*k22*n1*t2 - k1*k22*n2*t1 - k1*k22*r
                   * sqrt
                   + k1*k22*t1*t2 + k23*n22 - 2*k23*n2*t2
                   - k23*r2 + k23*t22)
                  / BCdenominator)},
        'D': {
            'x': k1 + k2*m.sqrt(-cos_half_theta**2 + 1.0)/cos_half_theta,
            'y': k2 - k1*m.sqrt(-cos_half_theta**2 + 1.0)/cos_half_theta}}

import math as m


cpdef main(
        double cam_pos_x,
        double cam_pos_y,
        double cam_k,
        double cam_theta,
        double cam_alpha,
        double obs_pos_x,
        double obs_pos_y,
        double obs_vel_x,
        double obs_vel_y,
        double obs_radius):
    """ It makes the image parameters into an int between 0 and 100. """

    # The width of the camera lens.
    cdef double z
    z = 2 * cam_k + m.tan(cam_theta / 2)

    err, parameters = obstacle_image_parameters(
        cam_pos_x,
        cam_pos_y,
        cam_k,
        cam_theta,
        cam_alpha,
        obs_pos_x,
        obs_pos_y,
        obs_vel_x,
        obs_vel_y,
        obs_radius)

    if err is not None:
        return err, None

    return (None, {
        'x': int(parameters['x'] * 100 / z),
        'y': int(parameters['y'] * 100 / z)})


cdef obstacle_image_parameters(
        double cam_pos_x,
        double cam_pos_y,
        double cam_k,
        double cam_theta,
        double cam_alpha,
        double obs_pos_x,
        double obs_pos_y,
        double obs_vel_x,
        double obs_vel_y,
        double obs_radius):
    """
    It takes in the position and orientation of a camera, and the
    position of an obstacle and calculates the parameters needed to
    construct the image in the camera.

    A diagram is shown in ./simulateTrig.pdf.  The required values are
    x and y (shown on the diagram).
    """
    err, points = calculate_ABCD_coords(
        cam_pos_x,
        cam_pos_y,
        cam_k,
        cam_theta,
        cam_alpha,
        obs_pos_x,
        obs_pos_y,
        obs_vel_x,
        obs_vel_y,
        obs_radius)
    if err is not None:
        return err, None
    X = flatten_points(
        points['A']['x'],
        points['A']['y'],
        points['B']['x'],
        points['B']['y'],
        points['C']['x'],
        points['C']['y'],
        points['D']['x'],
        points['D']['y'])
    cdef double A, B, C, D
    A = X['A']
    B = X['B']
    C = X['C']
    D = X['D']
    # The alternatives for when the obstacle is in view are:
    #     ABCD -> x = B - A, y = C - B
    #     BACD -> x = 0, y = C - A
    #     BADC -> x = 0, y = 100
    #     ABDC -> x = B - A, y = D - B
    if A <= B and B <= C and C <= D:
        return (None, {
            'x': B - A,
            'y': C - B})
    if B <= A and A <= C and C <= D:
        return (None, {
            'x': 0,
            'y': C - A})
    if B <= A and A <= D and D <= C:
        return (None, {
            'x': 0,
            'y': 100})
    if A <= B and B <= D and D <= C:
        return (None, {
            'x': B - A,
            'y': D - B})
    return "Obstacle is out of sight.", None


cdef flatten_points(
        double Ax,
        double Ay,
        double Bx,
        double By,
        double Cx,
        double Cy,
        double Dx,
        double Dy):
    """
    A, B, C and D are dictionaries, each containing 'x' and 'y' fields.
    These describe 2D Cartesian coordinates.  A, B, C and D are all on
    one straight line.  This function reduces them to one dimension
    by treating the common line as a real number line with A at 0 and
    D on the positive side of A.
    """
    cdef double Bflat, Cflat, Dflat
    Bflat = compare_to_AD(Ax, Ay, Dx, Dy, Bx, By)
    Cflat = compare_to_AD(Ax, Ay, Dx, Dy, Cx, Cy)
    Dflat = compare_to_AD(Ax, Ay, Dx, Dy, Dx, Dy)
    if Dflat < 0:
        Bflat = -Bflat
        Cflat = -Cflat
        Dflat = -Dflat
    return {
        'A': 0.0,
        'B': Bflat,
        'C': Cflat,
        'D': Dflat}


cdef calculate_ABCD_coords(
        double cam_pos_x,
        double cam_pos_y,
        double cam_k,
        double cam_theta,
        double cam_alpha,
        double obs_pos_x,
        double obs_pos_y,
        double obs_vel_x,
        double obs_vel_y,
        double obs_radius):
    """
    It calculates the coordinates of the points A, B, C and D, which
    are points along the lens-line of the camera.  They are shown on
    the diagram in ./simulateTrig.pdf.
    """
    cdef double k1, k2
    # t1 = cam_pos_x
    # t2 = cam_pos_y
    # n1 = obs_pos_x
    # n2 = obs_pos_y
    k1 = cam_k * m.cos(cam_alpha)
    k2 = cam_k * m.sin(cam_alpha)
    # r = obs_radius
    rel2cam = solve_geometry(
        k1, k2, obs_pos_x, obs_pos_y, obs_radius, cam_pos_x, cam_pos_y, cam_theta)
    AL = rel2cam['A']
    BL = rel2cam['B']
    CL = rel2cam['C']
    DL = rel2cam['D']
    PL = rel2cam['P']
    QL = rel2cam['Q']
    cdef double BxP, QxC
    BxP = BL['x']*PL['y'] - BL['y']*PL['x']
    QxC = QL['x']*CL['y'] - QL['y']*CL['x']
    if BxP < 0 or QxC < 0:
        return "Obstacle is out of sight.", None
    return (None, {
        'A': {'x': AL['x'] + cam_pos_x,
              'y': AL['y'] + cam_pos_y}, 
        'B': {'x': BL['x'] + cam_pos_x,
              'y': BL['y'] + cam_pos_y},
        'C': {'x': CL['x'] + cam_pos_x,
              'y': CL['y'] + cam_pos_y},
        'D': {'x': DL['x'] + cam_pos_x,
              'y': DL['y'] + cam_pos_y}})


cdef double compare_to_AD(double Ax, double Ay, double Dx, double Dy, double Xx, double Xy):
    """
    Each of the inputs contains two coordinates describing a point in
    a 2D Cartesian coordinate system.  All three points are on the same
    line.

    The function calculates where point X is on the real number line
    defined by points A and D where A is at zero and D is on the
    positive side of A.

    Let the straight line be y = mx + c where m is the gradient and
    c is the y-intercept.  The method used here is to first translate
    the line downwards by c, then rotate clockwise about the origin by
    arctan(m).  This leaves the length of the line unchanged, but
    positions it on the x-axis.  Then the y-coordinates of A, B, C and
    D can be ignored and the x-coordinates are used as the number line.
    """
    if (Dx - Ax)**2 < 0.000001:
        # The line is vertical.
        return Xy - Ay
    cdef double gradient, angle, cosangle, sinangle
    gradient = (Dy - Ay) / (Dx - Ax)
    angle = m.atan(gradient)
    cosangle = m.cos(angle)
    sinangle = m.sin(angle)
    # The rotated point is found by multiplying it by the rotation
    # matrix:
    #
    # [rot11 rot12] [p1] = [rot11 * p1 + rot12 * p2]
    # [rot21 rot22] [p2]   [rot21 * p1 + rot22 * p2]
    #
    # The y-coordinate is thrown away, so the wanted result is
    #
    #     rot11 * p1 + rot12 * p2
    #
    # In this case:
    #
    #     rot11 = cos(ϴ)
    #     rot12 = sin(ϴ)
    return (cosangle * Xx + sinangle * Xy) - (cosangle * Ax + sinangle * Ay)


def solve_geometry(double k1, double k2, double n1, double n2, double r, double t1, double t2, double theta):
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

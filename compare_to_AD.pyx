import math as m

cpdef double main(double Ax, double Ay, double Dx, double Dy, double Xx, double Xy):
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

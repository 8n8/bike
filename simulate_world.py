import math as m
import numpy as np

def calculate_sensor_readings(world_state):
    """
    If an obstacle is twice as far away it will appear half the size
    in the image, so:
     
        (width in image) = C1 / distance of object
        (height in image) = C1 / distance of object

    All the obstacles are cylinders to make it easier: they look the
    same from all angles.  It is assumed that the cameras are high
    enough off the ground and the obstacles are tall enough that the
    obstacles always go from top to bottom of the images.  The only 
    thing that can change about an image of an object is its width,
    horizontal position and orientation.

    It is assumed that a camera is the size of geometric point and that
    all the cameras are at the same place.
    """
    return {
        'front cam' = 


def camera_properties(x, y, orientation):
    """
    It is assumed that the cameras are all attached at the same point
    on the frame of the bike.
    """
    return {
        'front': generic_cam(orientation, x, y),
        'left': generic_cam(orientation + m.pi/2, x, y),
        'rear': generic_cam(orientation + m.pi, x, y),
        'right': generic_cam(orientation - m.pi/2, x, y)}


def generic_cam(alpha, x, y): 
    return {
        'x': x,
        'y': y,
        'k': 0.1,
        'theta': m.pi/2,
        'alpha': alpha}


def calculate_images(obstacles):
    



def _obstacle_image_parameters(cam, obs):
    """
    It takes in the position and orientation of a camera, and the
    position of an obstacle and calculates the parameters needed to
    construct the image in the camera.

    A diagram is shown in ./simulateTrig.pdf.  The required values are
    x and y (shown on the diagram).
    """
    A, B, C, D = flatten_points(calculate_ABDC_coords(cam, obs))
    z = width_of_camera_lens(cam)
    x = 0
    if B > 0:
        x = min(z, B)
    y = 0
    if D - C < 0:
        y = min(z, D - B)
    else:
        y = min(z, C - B)
    return {
        'x': x,
        'y': y }


def width_of_camera_lens(cam):
    """
    It calculates the width of the camera lens.  See page 6 of 
    ./simulateTrig.pdf for details.
    """
    return 2 * cam['k'] * m.arctan(cam['theta'] / 2)


def calculate_ABCD_coords(cam, obs):
    """
    It calculates the coordinates of the points A, B, C and D, which 
    are points along the lens-line of the camera.  They are shown on
    the diagram in ./simulateTrig.pdf.
    """
    return (
        _calculate_A(cam),
        _calculate_B(cam, obs),
        _calculate_C(cams, obs),
        _calculate_D(cam))


def flatten_points(A, B, C, D):
    """
    A, B, C and D are dictionaries, each containing 'x' and 'y' fields.
    These describe 2D Cartesian coordinates.  A, B, C and D are all on
    one straight line.  This function reduces them to one dimension
    by treating the common line as a real number line with A at 0 and
    D on the positive side of A.
    """
    def flatten(point):
        return _compare_to_AD(Apoint, Dpoint, point)
    Bflat = flatten(Bpoint)
    Cflat = flatten(Cpoint)
    Dflat = flatten(Dpoint)
    return 0, Bflat, Cflat, Dflat


def _compare_to_AD(A, D, X):
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
    gradient = (D['y'] - A['y']) / (D['x'] - A['x'])
    y_intercept = A['y'] - gradient * A['x']
    angle = m.arctan(gradient)
    rotation = np.array([
        [m.cos(angle), m.sin(angle)],
        [-m.sin(angle), m.cos(angle)]])
    def flatten(p):
        return np.matmul(
            rotation,
            np.array([[p['x']], [p['y'] - y_intercept]]))[0][0]
    Anew, Xnew = _flipsign(flatten(A), flatten(D), flatten(X))
    return Xnew - Anew


def _flipsign(Aflat, Dflat, Xflat):
    """
    Aflat is supposed to be less than Dflat on the real number line
    defined by them.  Xflat is another point on the number line.  This
    function flips the sign if Aflat is in fact bigger than Dflat.
    """
    if Aflat < Dflat:
        return Aflat, Xflat
    else
        return -Aflat, -Xflat


def _calculate_A(cam):
    """
    It calculates the position of the left-hand side of the camera
    lens.  See diagram and workings in ./simulatorTrig.pdf.
    """
    p = k / (m.cos(cam['theta'] / 2))
    beta = m.pi - (cam['theta'] / 2) - cam['alpha']
    x1 = p * m.cos(beta)
    x2 = p * m.sin(beta)
    return {
        'x': cam['x'] - x1,
        'y': cam['y'] + x2 }


def _calculate_B(cam, obs):
    """
    It calculates the position of the intercept of the left-hand
    view-line of the obstacle and the lens line.  See diagram and
    workings in ./simulateTrig.pdf.
    """
    z = ((obs['x'] - cam['x'])**2 + (obs['y'] - cam['y'])**2)**0.5
    phi1 = m.arctan((obs['y'] - cam['y']) / (obs['x'] - cam['x']))
    phi2 = m.arcsin(obs['radius'] / z)
    phi = phi1 + phi2
    u = alpha - phi
    d1 = cam['k'] * m.tan(u)
    s = (k**2 + d1**2)**0.5
    x2 = s * m.cos(cam['alpha'])
    x3 = s * m.sin(cam['alpha'])
    return {
        'x': cam['x'] + x2,
        'y': cam['y'] + x3 }


def _calculate_C(cam, obs):
    """
    It calculates the position of the intercept of the right-hand
    view-line of the obstacle and the lens line.  See diagram and
    workings in ./simulateTrig.pdf.
    """
    x1 = ((obs['x'] - cam['x'])**2 + (obs['y'] - cam['y'])**2)**0.5
    x2 = (x1**2 - obs['radius']**2)**0.5
    phi2 = m.arcsin(obs['radius'] / x2)
    x5 = obs['y'] - cam['y'] 
    phi5 = m.arcsin(x5 / x1)
    phi1 = phi5 - phi2
    phi3 = alpha - phi5
    phi4 = phi2 + phi3
    s = k / m.cos(phi4)
    x3 = s * m.cos(phi1)
    x4 = s * m.sin(phi1)
    return {
        'x': cam['x'] + x3,
        'y': cam['y'] + x4 }


def _calculate_D(c, d, k, theta, alpha):
    """
    It calculates the position of the right-hand end of the camera
    lens.
    """
    p = cam['k'] / m.cos(cam['theta'] / 2)
    beta = cam['alpha'] - (cam['theta'] / 2)
    a = p * m.cos(beta)
    b = p * m.sin(beta)
    return {
        'x': cam['x'] + a,
        'y': cam['y'] + b }

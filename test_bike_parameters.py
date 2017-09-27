import bike_parameters as b


def basic_parameters1() -> b.BasicParameters:
    return {
        'bike': {
            'wheelbase': 1.02,
            'trail': 0.08,
            'steertilt': 0.314},
        'back_wheel': {
            'radius': 0.3,
            'mass': 2,
            # The mass moment of inertia about an axis
            # in the perpendicular plane.
            'ixx': 0.0603,
            # The mass moment of inertia about the axle.
            'iyy': 0.12},
        'main_frame': {
            # The x and z coordinates of the centre of mass.
            # The origin is the contact point of the back wheel.
            'comx': 0.3,
            'comz': -0.9,
            'mass': 85,
            'ixx': 9.2,
            'ixz': 2.4,
            'iyy': 11,
            'izz': 2.8},
        'front_fork': {
            'comx': 0.9,
            'comz': -0.7,
            'mass': 4,
            'ixx': 0.05892,
            'ixz': -0.00756,
            'iyy': 0.06,
            'izz': 0.00708},
        'front wheel': {
            'radius': 0.35,
            'mass': 3,
            # The mass moment of inertia about an axis
            # in the perpendicular plane.
            'ixx': 0.1405,
            # The mass moment of inertia about the axle.
            'iyy': 0.28}}


def test_make():
    # It is hard to write a meaningful test for this function
    # without simply duplicating the logic.  This assersion simply
    # runs the function, so at least it will find runtime errors.
    assert b.make(basic_parameters1()) != None

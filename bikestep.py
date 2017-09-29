"""
It exports one function (main) that increments the bike state by one
step, using the equations of motion taken from ref meijaard07 in the
docs.

The documentation for this module is in
bikeproject/simulateWorld/bikestep/main.tex.
"""


import math as m
import world2sensor as w
from mypy_extensions import TypedDict
from scipy.integrate import odeint  # type: ignore
from typing import List


# The matrix elements for the bicycle equation of motion.  These are
# calculated from the parameters of the bicycle, such as its mass and
# the size of its wheels.  These are taken from ref meijaard07 in the
# docs.
m1: float = 80.817_22
m2: float = 2.319_413_322_087_09
m3: float = m2
m4: float = 0.297_841_881_996_86
c1: float = 0
c2: float = 33.866_413_914_924_94
c3: float = -0.850_356_414_569_78
c4: float = 1.685_403_973_975_60
h1: float = -794.1195  # 9.81 x -80.95
h2: float = -25.501_260_32  # 9.81 x -2.599,516,852,498,72
h3: float = h2
h4: float = -7.880_322_818  # 9.81 x -0.803,294,884,586,18
k1: float = 0
k2: float = 76.597_345_895_732_22
k3: float = 0
k4: float = 2.654_315_237_946_04

m2xm3_m1xm4: float = m2 * m3 - m1 * m4


def _p2dot(b: w.BikeState) -> float:
    """
    This equation is the one for dp_2/dt taken from the main
    documentation for this module.  Briefly, it is one of the
    set of first-order odes created from the main bike equation of
    motion in ref meijaard07.  The variable p_2 is the same as
    d phi / dt.
    """
    return (
        -(-b['delta'] * h2 * m4 + (h3 * m2 - h1 * m4) * b['phi'] + m2
          * (b['delta'] * h4 - b['Tdelta']) + ((c3 * m2 - c1 * m4)
          * b['phidot'] - c2 * b['deltadot'] * m4 + c4 * b['deltadot']
          * m2) * b['v'] + ((k3 * m2 - k1 * m4) * b['phi'] - b['delta']
          * k2 * m4 + b['delta'] * k4 * m2) * b['v']**2)
        / m2xm3_m1xm4)


def _d2dot(b: w.BikeState) -> float:
    """
    This equation is the one for dd_2/dt taken from the main docs
    for this module.  d_2 is the same as d delta / dt.
    """
    return (
        (-b['delta'] * h2 * m3 + (h3 * m1 - h1 * m3) * b['phi'] + m1
         * (b['delta'] * h4 - b['Tdelta']) +
         ((c3 * m1 - c1 * m3) * b['phidot'] - c2 * b['deltadot'] * m3
          + c4 * b['deltadot'] * m1)
         * b['v'] + ((k3 * m1 - k1 * m3) * b['phi'] - b['delta']
         * k2 * m3 + b['delta'] * k4 * m1) * b['v']**2)
        / m2xm3_m1xm4)


# It represents the set of first-order derivatives that are later
# on fed into the ode solver.
class Derivatives(TypedDict):
    phidot: float
    deltadot: float
    p2dot: float
    d2dot: float
    psidot: float
    xdot: float
    ydot: float


def _derivatives2list(d: Derivatives) -> List[float]:
    """
    It converts the dictionary containing the set of first-order
    derivatives into the list format required by the scipy module's
    ode solver, which is used later on.
    """
    return [
        d['phidot'],
        d['deltadot'],
        d['p2dot'],
        d['d2dot'],
        d['psidot'],
        d['xdot'],
        d['ydot']]


def _list2bikestate(L: List['float'], v, Tdelta, Tm) -> w.BikeState:
    """
    A converter from a list representing most of the bike state to
    the bikestate dictionary.  This is necessary because the scipy
    module's ode solver used later only works with lists.
    """
    return {
        'phi': L[0],
        'delta': L[1],
        'phidot': L[2],
        'deltadot': L[3],
        'psi': L[4],
        'x': L[5],
        'y': L[6],
        'v': v,
        'Tdelta': Tdelta,
        'Tm': Tm}


def _calculate_velocity(Tm: float, t: float, v0: float) -> float:
    """
    The velocity equation (see the docs) is:

            Tm . t
        v = ------ + v0
            rF . m

    where:

        Tm     drive torque applied at front wheel hub
        rF     radius of front wheel
        m      total mass of the bike
        v0     initial velocity of bike

    In this case (see Table 1 in ref meijaard07 in the docs):

        rF = 0.35 metres
        m  = 2 + 85 + 4 + 3 = 94 kg *

    so

        r_F . m = 0.35 x 94 = 32.9

    * Note that the reason the mass is so high is that the model
    includes a large point mass to represent the rider.
    """
    return ((Tm * t) / 32.9) + v0


def _derivatives(b: w.BikeState, t: float) -> Derivatives:
    """
    It calculates a set of six first-order derivatives.  The
    equations for these derivatives are derived from the bike
    equations of motion in the ref meijaard07.  Look in the docs
    for the full derivation.  These derivatives are required in this
    form by the scipy module's ode solver, which is used later.
    """
    cos_lambda: float = 0.951_056_516_3  # cos(pi/10)
    # The steer axis tilt.
    c: float = 0.08
    # The distance between the wheel contact points.
    w: float = 1.02
    return {
        'phidot': b['phidot'],
        'deltadot': b['deltadot'],
        'p2dot': _p2dot(b),
        'd2dot': _d2dot(b),
        'psidot': (
            ((b['v'] * b['delta'] + c * b['deltadot']) / w)
            * cos_lambda),
        'xdot': b['v'] * m.cos(b['psi']),
        'ydot': b['v'] * m.sin(b['psi'])}


def _derivList(
        L: List[float],
        t: float,
        v: float,
        Tdelta: float,
        Tm: float
        ) -> List[float]:
    """
    It takes in a list of the bike parameters and calculates the
    gradients at that point.  The scipy module's ode solver requires
    a function in this specific form for its first input.
    """
    b: w.BikeState = _list2bikestate(L, v, Tdelta, Tm)
    derivs: Derivatives = _derivatives(b, t)
    return _derivatives2list(derivs)


def _make_initial_conditions(b: w.BikeState) -> List[float]:
    """
    It extracts the initial conditions in the form of a list of
    floats, using the dictionary containing the bike state.  A list
    format is required for the scipy ode solver module.
    """
    return [
        b['phi'],
        b['delta'],
        b['phidot'],
        b['deltadot'],
        b['psi'],
        b['x'],
        b['y']]


def main(b: w.BikeState, t: float) -> w.BikeState:
    """
    This is the only function exported by this module.  Given the
    current state of the bicycle and a time period, it calculates
    the state of the bicycle at the end of the time period.
    """
    v: float = _calculate_velocity(b['Tm'], t, b['v'])
    sol = odeint(  # type: ignore
        _derivList,
        _make_initial_conditions(b),
        [0, t],
        args=(v, b['Tdelta'], b['Tm']))[1]
    return {
        'phi': sol[0],
        'delta': sol[1],
        'phidot': sol[2],
        'deltadot': sol[3],
        'psi': sol[4],
        'x': sol[5],
        'y': sol[6],
        'v': v,
        'Tdelta': b['Tdelta'],
        'Tm': b['Tm']}

import math as m
import world2sensor as w
from mypy_extensions import TypedDict
from scipy.integrate import odeint  # type: ignore
from typing import List

"""
The documentation for this module is in
bikeproject/simulateWorld/bikestep/main.tex.
"""

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


def p2dot(b: w.BikeState) -> float:
    return (
        -(-b['delta'] * h2 * m4 + (h3 * m2 - h1 * m4) * b['phi'] + m2
          * (b['delta'] * h4 - b['Tdelta']) + ((c3 * m2 - c1 * m4)
          * b['phidot'] - c2 * b['deltadot'] * m4 + c4 * b['deltadot']
          * m2) * b['v'] + ((k3 * m2 - k1 * m4) * b['phi'] - b['delta']
          * k2 * m4 + b['delta'] * k4 * m2) * b['v']**2)
        / m2xm3_m1xm4)


def d2dot(b: w.BikeState) -> float:
    return (
        (-b['delta'] * h2 * m3 + (h3 * m1 - h1 * m3) * b['phi'] + m1
         * (b['delta'] * h4 - b['Tdelta']) +
         ((c3 * m1 - c1 * m3) * b['phidot'] - c2 * b['deltadot'] * m3
          + c4 * b['deltadot'] * m1)
         * b['v'] + ((k3 * m1 - k1 * m3) * b['phi'] - b['delta']
         * k2 * m3 + b['delta'] * k4 * m1) * b['v']**2)
        / m2xm3_m1xm4)


class Derivatives(TypedDict):
    phidot: float
    deltadot: float
    p2dot: float
    d2dot: float
    psidot: float
    xdot: float
    ydot: float


def derivatives2list(d: Derivatives) -> List[float]:
    return [
        d['phidot'],
        d['deltadot'],
        d['p2dot'],
        d['d2dot'],
        d['psidot'],
        d['xdot'],
        d['ydot']]


def calculate_velocity(Tm: float, t: float, v0: float) -> float:
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
        m  = 2 + 85 + 4 + 3 = 94 kg

    so

        r_F . m = 0.35 x 94 = 32.9
    """
    return ((Tm * t) / 32.9) + v0


def derivatives(b: w.BikeState, t: float) -> Derivatives:
    cos_lambda: float = 0.951_056_516_3  # cos(pi/10)
    # The steer axis tilt.
    c: float = 0.08
    # The distance between the wheel contact points.
    w: float = 1.02
    return {
        'phidot': b['phidot'],
        'deltadot': b['deltadot'],
        'p2dot': p2dot(b),
        'd2dot': d2dot(b),
        'psidot': (
            ((b['v'] * b['delta'] + c * b['deltadot']) / w)
            * cos_lambda),
        'xdot': b['v'] * m.cos(b['psi']),
        'ydot': b['v'] * m.sin(b['psi'])}


def make_initial_conditions(b: w.BikeState) -> List[float]:
    return [
        b['phi'],
        b['delta'],
        b['phidot'],
        b['deltadot'],
        b['psi'],
        b['x'],
        b['y']]


def main(b: w.BikeState, t: float) -> w.BikeState:
    sol: List[float] = odeint(
        derivatives2list(derivatives(b, t)),
        make_initial_conditions(b),
        t)
    return {
        'phi': sol[0],
        'delta': sol[1],
        'phidot': sol[2],
        'deltadot': sol[3],
        'psi': sol[4],
        'x': sol[5],
        'y': sol[6],
        'v': calculate_velocity(b['Tm'], t, b['v']),
        'Tdelta': b['Tdelta'],
        'Tm': b['Tm']}

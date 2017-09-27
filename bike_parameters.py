from mypy_extensions import TypedDict


class Bike(TypedDict):
    wheelbase: float
    trail: float
    steertilt: float


class BackWheel(TypedDict):
    radius: float
    mass: float
    ixx: float
    iyy: float


class MainFrame(TypedDict):
    comx: float
    comz: float
    mass: float
    ixx: float
    ixz: float
    iyy: float
    izz: float


class FrontFork(TypedDict):
    comx: float
    comz: float
    mass: float
    ixx: float
    ixz: float
    iyy: float
    izz: float


class FrontWheel(TypedDict):
    radius: float
    mass: float
    ixx: float
    iyy: float


class BasicParameters(TypedDict):
    bike: Bike
    back_wheel: BackWheel
    main_frame: MainFrame
    front_fork: FrontFork
    front_wheel: FrontWheel


class DerivedParameters(TypedDict):
    m1: float
    m2: float
    m3: float
    m4: float
    c1: float
    c2: float
    c3: float
    c4: float
    h1: float
    h2: float
    h3: float
    h4: float
    k1: float
    k2: float
    k3: float
    k4: float


def make(b: BasicParameters) -> DerivedParameters:
    """
    It calculates the matrix constant terms in the bike equation of
    motion.  Look in the docs at 
    bikeproject/simulateWorld/bikestep/main.tex for the details.
    """
    IRxx = b['back_wheel']['ixx'] 
    ITxx = (IRxx + IBxx + IHxx + IFxx + mR * rR**2 + mB * zB**2
        + mH * zH**2 + mF * rF**2)
    m1 = ITxx 

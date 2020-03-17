import math

class Consts:
    rEarth = 6371009  # Earth radius, [m]
    rMars = 3396e3  # Mars radius, [m]
    rEarth_equatorial = 6.378136300e6  # mean Earth equtorial radius, [m]
    muEarth = 3.986004415e14  # Earth stadard gravitational parameter, [m^3 / s^2]
    muSun = 132712440017.987 * 10 ** 9  # Sun gravitational parameter, [m3/s2]
    AstronomicUnit = 149597870691  # AstronomicUnit, [m]
    EarthMeanMotion = math.sqrt(muSun / AstronomicUnit ** 3)  # mean motion of the Earth, [rad/s]
    J2 = 1.082626e-3  # First zonal harmonic coefficient in the expansion of the Earth's gravity field
    deg2rad = math.pi / 180  # conversion from degrees to radians
    rad2deg = 180 / math.pi  # conversion from radians to degrees
    km2m = 1000  # conversion from kilometers to meters
    m2km = 1e-3  # conversion from meters to kilometers
    omegaEarth = 7.29211585275553e-005  # Earth self revolution angular vecocity [rad/s]
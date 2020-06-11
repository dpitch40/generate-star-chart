import math
import random
import csv
import argparse
import operator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

direction_mapping = dict(zip(range(0, 361, 15),
                             [f'{i}h' for i in range(24)]))

with open('BlackBody.csv', 'r') as fobj:
    # See http://www.vendian.org/mncharity/dir3/blackbody/
    reader = csv.DictReader(fobj)
    color_temp = list()
    for row in reader:
        row['Temperature'] = int(row['Temperature'])
        color_temp.append(row)

# Utility

def append_degree(x, pos):
    return f'{x}°'

def bol_corr(T):
    # Estimate of a function for calculation bolometric correction
    return -3e-7 * (T - 6000) ** 2 - 0.1 if T < 6000 else \
           -1.6e-7 * (T - 6000) ** 1.6 - 0.1

def distance(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

def gaussian(x, alpha, mu, sigma1, sigma2):
    sqrt = (x - mu) / (sigma1 if x < mu else sigma2)
    return alpha * math.exp(- (sqrt ** 2) / 2)

def luminosity_to_mbol(L):
    return -2.5 * math.log10((L * 3.828e26) / 3.0128e28)

def power_interpolate(x, thresholds, powers, scale=1):
    """Creates a piecewise power function with varying exponents and automatic scaling of the
       pieces so the function is continuous.

       x    The value to run through the function
       thresholds   List of x-values corresponding to divisions between the pieces
       powers   List of exponents to use for each piece. Should be 1 longer than thresholds.
       scale    The constant coefficient of the first function piece. The others will be
                inferred from it to keep the function continuous.
    """
    coeffs = [scale]
    for i, t in enumerate(thresholds):
        last = coeffs[-1]
        last_power = powers[i]
        if x < t:
            return last * x ** last_power

        power = powers[i+1]
        coeffs.append(((last * t ** last_power) / t ** power))
    else:
        return coeffs[-1] * x ** powers[-1]

def temp_to_color(T):
    for row in color_temp:
        if T < row['Temperature']:
            return row['Hex']
    return color_temp[-1]['Hex']

class Star:
    __slots__ = ['mass', 'L', 'T', 'x', 'y', 'z', 'distance', 'dec', 'ra',
                 'Mbol', 'Mabs', 'Mapp', 'color']
    pc = 3.26156

    @classmethod
    def from_luminosity(cls, mass, T, x, y, z, L):
        inst = super().__new__(cls)
        inst.init_basic(mass, T, x, y, z)
        inst.L = L
        inst.init_absolute_magnitude()
        inst.init_apparent_magnitude()
        inst.init_angular_pos()
        inst.init_color()
        return inst

    @classmethod
    def from_mabs(cls, mass, T, x, y, z, Mabs):
        inst = super().__new__(cls)
        inst.init_basic(mass, T, x, y, z)
        inst.Mabs = Mabs
        inst.init_apparent_magnitude()
        inst.init_angular_pos()
        inst.init_color()
        return inst

    def init_absolute_magnitude(self):
        self.Mbol = luminosity_to_mbol(self.L)
        self.Mabs = self.Mbol - bol_corr(self.T)

    def init_angular_pos(self):
        self.ra = math.degrees(math.atan2(self.y, self.x))
        self.dec = math.degrees(math.asin(self.z / self.distance))

    def init_apparent_magnitude(self):
        self.Mapp = self.Mabs + 5 * math.log10(self.distance / self.pc) - 5

    def init_basic(self, mass, T, x, y, z):
        self.mass = mass
        self.T = T
        self.x = x
        self.y = y
        self.z = z
        self.distance = distance(self.x, self.y, self.z, 0, 0, 0)

    def init_color(self):
        self.color = temp_to_color(self.T)

    @staticmethod
    def mag_to_size(mag):
        return (1.5 + 1 / (1 + math.exp(-0.5 * (mag + 4)))) ** (4 - mag)

    def to_array_row(self):
        return [self.dec, self.ra, self.mag_to_size(self.Mapp), self.color]

class StarForge:
    min_mass = 0.05
    max_mass = 300

    area_1 = (25 / 0.7) * (0.08 ** 0.7)
    area_2 = (2 / 0.3) * (0.08 ** -0.3 - 0.5 ** -0.3)

    max_radius = 5000

    @staticmethod
    def imf(m):
        # Initial mass function (PDF) from Kroupa (2001), takes a mass in solar
        # masses and returns probability density
        # See https://arxiv.org/pdf/astro-ph/0009005.pdf
        if m >= 0.5:
            return m ** -2.3
        elif m >= 0.08:
            return (m ** -1.3) * 2
        else:
            return (m ** -0.3) * 25

    @classmethod
    def int_imf(cls, m):
        # Definite integral of the IMF from 0 to m
        if m < 0.08:
            return (25 / 0.7) * (m ** 0.7)
        elif m < 0.5:
            return cls.area_1 + (2 / 0.3) * (0.08 ** -0.3 - m ** -0.3)
        else:
            return cls.area_1 + cls.area_2 + (1 / 1.3) * (0.5 ** -1.3 - m ** -1.3)

    @classmethod
    def random_star_mass(cls, min_mass=None, max_mass=None):
        if min_mass is None:
            min_mass = cls.min_mass
        if max_mass is None:
            max_mass = cls.max_mass
        min_area = cls.int_imf(min_mass)
        max_area = cls.int_imf(max_mass)

        area = random.uniform(min_area, max_area)
        if area < cls.area_1:
            # mass < 0.08
            return (area * (0.7 / 25)) ** (1 / 0.7)
        elif area < cls.area_1 + cls.area_2:
            # 0.08 <= mass < 0.5
            return (0.08 ** -0.3 - (area - cls.area_1) * (0.3 / 2)) ** (1 / -0.3)
        else:
            # 0.5 <= mass

            return (0.5 ** -1.3 - (area - cls.area_1 - cls.area_2) * 1.3) ** (1 / -1.3)

    @staticmethod
    def star_lifetime(m):
        # Simplistic model: m ^ -2 for m < 1, m ^ -2.5 for m > 1, times 10 billion
        if m < 1:
            exp = -2
        else:
            exp = -2.5

        return 1e10 * m ** exp

    @staticmethod
    def ms_luminosity(m):
        # Luminosity approximation for stars on the main sequence
        # In solar luminosities
        # See https://en.wikipedia.org/wiki/Mass%E2%80%93luminosity_relation
        return power_interpolate(m, [0.43, 2, 55], [2.3, 4, 3.5, 1], 0.23)

    @staticmethod
    def ms_temperature(m):
        # Temoerature approximation for main sequence stars
        # See https://en.wikipedia.org/wiki/Stellar_classification#Harvard_spectral_classification
        return power_interpolate(m, [0.5, 2], [0.3, 0.65, 0.55], 4800)

    @staticmethod
    def giant_luminosity(m):
        return 2800 * m ** 0.1

    @staticmethod
    def giant_temperature(m):
        return power_interpolate(m, [1.0, 2.0], [0.16, 0.5, -0.1], 3150)

    @classmethod
    def supergiant_magnitude(cls, m, T):
        # Easier to calculate magnitude
        lower_bound = cls.ms_temperature(m / 30)
        Mabs = -3 - math.log10((T - lower_bound))
        return Mabs

    @classmethod
    def supergiant_temperature(cls, m):
        # Very handwavy
        base_t = cls.ms_temperature(m) * 1.1
        lower_bound = cls.ms_temperature(m / 30)
        mode = lower_bound + (base_t - lower_bound) * (m / 75)
        return random.triangular(lower_bound, base_t, mode)

    @classmethod
    def gen_main_sequence_star(cls, mass):
        L = cls.ms_luminosity(mass)
        T = cls.ms_temperature(mass)
        return L, T

    @classmethod
    def random_star_pos(cls, max_radius=None, min_radius=0):
        if max_radius is None:
            max_radius = cls.max_radius
        x, y, z = cls.max_radius, cls.max_radius, cls.max_radius
        d = distance(x, y, z, 0, 0, 0)
        while d > max_radius or d < min_radius:
            x = random.uniform(-cls.max_radius, cls.max_radius)
            y = random.uniform(-cls.max_radius, cls.max_radius)
            z = random.uniform(-cls.max_radius, cls.max_radius)
            d = distance(x, y, z, 0, 0, 0)
        return x, y, z

    @classmethod
    def random_star(cls, min_mass=None, max_mass=None, min_radius=0, max_radius=None):
        x, y, z = cls.random_star_pos(max_radius, min_radius)

        mass = cls.random_star_mass(min_mass, max_mass)
        L, T = cls.gen_main_sequence_star(mass)
        lifetime = cls.star_lifetime(mass)

        if mass > 8 and mass < 75: # Chance of supergiant
            # Very simplified
            is_supergiant = random.random() < 0.05
            if is_supergiant:
                T = cls.supergiant_temperature(mass)
                Mabs = cls.supergiant_magnitude(mass, T)
                return Star.from_mabs(mass, T, x, y, z, Mabs)
        elif mass > 0.3: # Chance of red giant
            # Very crude approximation
            rgb_time = lifetime * 0.05 / mass
            r = random.uniform(0, lifetime)
            is_giant = r < rgb_time
            if is_giant:
                giant_L = cls.giant_luminosity(mass)
                giant_T = cls.giant_temperature(mass)
                how_giant = (r / rgb_time) ** 0.5
                L = L + how_giant * (giant_L - L)
                T = T + how_giant * (giant_T - T)

        return Star.from_luminosity(mass, T, x, y, z, L)


# Star chart generating classes

class Chart:
    def plot_stars(self, stars, colors):
        data = {'lat': self.transform_lat(stars[:, 0]),
                'lon': self.transform_lon(stars[:, 1]),
                'size': stars[:, 2],
                'color': colors}
        self.ax.scatter('lon', 'lat', s='size', c='color', data=data)

    def transform_lat(self, lat):
        return lat

    def transform_lat(self, lon):
        return lon

    def save(self):
        self.ax.set_facecolor('black')
        self.fig.savefig(self.name + '.png', bbox_inches='tight', pad_inches=0.05)

class PolarChart(Chart):
    def __init__(self, name, center_lat, perim_lat, reverse=False):
        self.name = name
        self.reverse = reverse
        self.fig = plt.figure(figsize=(20, 20))
        self.ax = self.fig.add_subplot(projection='polar')
        self.ax.set_ylim(center_lat, perim_lat)
        self.center_lat = center_lat
        self.perim_lat = perim_lat

        self.ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 2 * math.pi, math.pi / 6)))
        self.ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(0, 2 * math.pi, math.pi / 12)))
        self.ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.formatter))
        self.ax.xaxis.set_minor_formatter(ticker.FuncFormatter(self.formatter))
        diff = 1 if perim_lat > center_lat else -1
        self.ax.yaxis.set_major_formatter(ticker.FuncFormatter(append_degree))
        self.ax.yaxis.set_major_locator(ticker.FixedLocator(range(center_lat+10*diff, perim_lat-diff, 10*diff)))
        self.ax.yaxis.set_minor_locator(ticker.FixedLocator(range(center_lat+5*diff, perim_lat-diff, 5*diff)))
        self.ax.tick_params(which='both', axis='y', labelcolor='#ffffff')
        self.ax.grid(which='major', axis='both', color='#ffffff')
        self.ax.grid(which='minor', axis='both', color='#808080')

    def formatter(self, x, pos):
        if self.reverse:
            x = 2 * math.pi - x
        return direction_mapping.get(round(math.degrees(x)) % 360, x)

    def transform_lon(self, lon):
        lon = lon * math.pi / 180
        if self.reverse:
            lon = 2 * math.pi - lon
        return lon

class RectChart(Chart):
    def __init__(self, name):
        self.name = name
        self.fig = plt.figure(figsize=(36, 10))
        self.ax = self.fig.add_subplot()
        self.ax.set_xlim(360, 0)
        self.ax.set_ylim(-50, 50)
        self.ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 361, 30)))
        self.ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(0, 361, 15)))
        self.ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.formatter))
        self.ax.xaxis.set_minor_formatter(ticker.FuncFormatter(self.formatter))
        self.ax.yaxis.set_major_locator(ticker.FixedLocator(range(-40, 50, 10)))
        self.ax.yaxis.set_minor_locator(ticker.FixedLocator(range(-45, 50, 5)))
        self.ax.yaxis.set_major_formatter(ticker.FuncFormatter(append_degree))
        self.ax.grid(which='major', axis='both', color='#ffffff')
        self.ax.grid(which='minor', axis='both', color='#808080')

    def formatter(self, x, pos):
        return direction_mapping.get((x - 180) % 360, x)

    def transform_lon(self, lon):
        return lon + 180

def generate_stars(args):
    stars = list()

    while len(stars) < args.n:
        star = StarForge.random_star(max_radius=args.radius)
        if star.Mapp < 6:
            stars.append(star)

    return stars

def save_stars(stars):
    with open('Stars.csv', 'w') as fobj:
        writer = csv.writer(fobj)
        writer.writerow(['Mass', 'Temp', 'Distance', 'Dec', 'RA', 'Mabs', 'Mapp'])
        for s in sorted(stars, key=operator.attrgetter('Mapp')):
            writer.writerow([round(s.mass, 2), int(s.T), round(s.distance, 2),
                             round(s.dec, 2), round(s.ra, 2), round(s.Mabs, 4), round(s.Mapp, 4)])

def chart_stars(stars):
    star_info = list()
    colors = list()
    for star in stars:
        dec, ra, scale, color = star.to_array_row()
        star_info.append([dec, ra, scale])
        colors.append(color)

    star_info = np.array(star_info)
    colors = np.array(colors)

    north = PolarChart('North', 90, 40, True)
    south = PolarChart('South', -90, -40)
    center = RectChart('Center')

    for chart in (north, south, center):
        chart.plot_stars(star_info, colors)
        chart.save()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--radius', type=int, default=5000,
                        help='The radius (in light years) within which stars will be generated')
    parser.add_argument('n', type=int, help='The number of visible stars to generate')
    args = parser.parse_args()

    stars = generate_stars(args)
    save_stars(stars)
    chart_stars(stars)

if __name__ == '__main__':
    main()

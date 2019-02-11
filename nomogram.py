from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

# gravitational acceleration
g = 9.81  # m/s²

# kinematic viscosity
ny = 1.3e-6  # m^2/s (10°C water)


# _________________________________________________________________________________________________________________
def log_scale(start, end, minor=False, lower=None, upper=None):
    """
    get the log scale ticks for the diagram

    Args:
        start (int):
        end (int):
        minor (bool):
        lower (int | float):
        upper (int | float):

    Returns:
        numpy.array: ticks of the scale
    """
    if minor:
        std = np.array([1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.,
                        2.2, 2.4, 2.6, 2.8, 3., 3.2, 3.4, 3.6, 3.8, 4., 4.2,
                        4.4, 4.6, 4.8, 5., 5.5, 6., 6.5, 7., 7.5, 8., 8.5,
                        9., 9.5, 10.])
    else:
        std = np.array([1., 1.5, 2., 3., 4., 5., 6., 8., 10.])

    res = np.array([])
    for x in range(start, end):
        res = np.append(res, std * 10. ** x)

    res = np.unique(res.round(3))

    if lower is not None:
        res = res[res >= lower]

    if upper is not None:
        res = res[res <= upper]

    return res


def nomogram(k=0.1):
    """
    make the nomogram

    Args:
        k (float): roughness in (mm)

    Returns:
        matplotlib.pyplot.Figure: of the plot
    """

    # diameter
    d = np.array(
        [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
         1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])  # m

    # velocity
    v = np.array(
        [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
         6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])  # m/s

    # head loss
    J = log_scale(-1, 3, minor=True)  # mm/m
    J_labels = log_scale(-1, 3, minor=False)

    # flow
    Q = log_scale(-1, 5, minor=True, upper=20000)  # L/s
    Q_labels = log_scale(-1, 5, minor=False, upper=20000)

    # _________________________________________________________________________________________________________________
    def area(d):
        return d ** 2 * pi / 4

    # _________________________________________________________________________________________________________________
    def velocity(J, d):
        return -2 * np.log10(2.51 * ny / (d * np.sqrt(2 * g * (J / 1000) * d)) +
                             (k / 1000) / (3.71 * d)) * \
               np.sqrt(2 * g * d * (J / 1000))

    # _________________________________________________________________________________________________________________
    def get_diameter(v, J):
        res = minimize_scalar(lambda x: abs(velocity(J, x) - v), bounds=(min(d), max(d)), method='bounded').x
        if (round(res, 5) >= max(d)) or (round(res, 5) <= min(d)):
            return np.NaN
        return res

    # _________________________________________________________________________________________________________________
    fig, ax = plt.subplots()

    def bbox(pad):
        return {'facecolor': 'white', 'alpha': 0.8, 'pad': pad, 'linewidth': 0}

    # _________________________________________________________________________________________________________________
    # diameter lines
    df_d = pd.DataFrame(index=J, columns=d)
    first = True
    for d_ in df_d:
        vi = velocity(df_d.index.values, d_)
        df_d[d_] = area(d_) * vi * 1000

        # change_d = 0.6
        # low, up = [0.34, 5.4]
        change_d = np.NaN
        low, up = [2.2, 2.2]

        if d_ == change_d:
            tvs = [low, up]
        elif d_ < change_d:
            tvs = [low]
        else:
            tvs = [up]

        for tv in tvs:
            tx = np.interp(tv, vi, J)
            ty = area(d_) * tv * 1000

            if first or d_ in (change_d, max(d)):
                txt = 'd={}m'.format(d_)
                if first:
                    first = False
            else:
                txt = d_

            ax.text(tx, ty, txt, fontsize=5, rotation=30, horizontalalignment='center', verticalalignment='bottom',
                    bbox=bbox(1))

    ax = df_d.plot(c='black', legend=False, logy=True, logx=True, ax=ax, lw=0.5)

    # _________________________________________________________________________________________________________________
    # velocity lines
    print('0')
    df_v = pd.DataFrame(index=np.logspace(-1, 3, num=500), columns=v)
    # df_v = pd.DataFrame(index=J, columns=v)

    first = True
    for v_ in df_v:
        d_ = df_v.index.to_series().apply(lambda Ji: get_diameter(v_, Ji)).values
        # d_ = np.array([get_d(v_, Ji) for Ji in df_v.index.values])
        Ai = area(d_)
        df_v[v_] = Ai * v_ * 1000

        # change_v = 5.
        # low, up = [0.043, 0.43]
        change_v = 9.
        low, up = [0.11, 0.43]

        if v_ == change_v:
            tds = [low, up]
        elif v_ < change_v:
            tds = [low]
        else:
            tds = [up]

        for td in tds:

            data = pd.DataFrame()
            data['d'] = d_
            data['J'] = df_v.index.values
            data.dropna(inplace=True)
            data.sort_values('d', inplace=True)

            tx = np.interp(td, data['d'].values, data['J'].values)
            ty = area(td) * v_ * 1000

            if first or (v_ in (change_v, max(v))):
                txt = 'v={}m/s'.format(v_).replace('.0', '')
                if first:
                    first = False
            else:
                txt = v_

            if pd.notna(tx) and pd.notna(ty):
                ax.text(tx, ty, txt, fontsize=5, rotation=-60, horizontalalignment='center', verticalalignment='bottom',
                        bbox=bbox(1))

    print('1')
    ax = df_v.plot(c='black', legend=False, logy=True, logx=True, ax=ax, lw=0.5)

    # _________________________________________________________________________________________________________________
    ax.set_xticks(J, minor=True)
    ax.set_yticks(Q, minor=True)

    ax.set_xticks(J_labels, minor=False)
    ax.set_yticks(Q_labels, minor=False)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_xticklabels([], minor=True)
    ax.set_yticklabels([], minor=True)

    ax.set_xticklabels([str(x).replace('.00', '').replace('.0', '') for x in J_labels], fontsize=6,
                       fontstretch='ultra-condensed')
    ax.set_yticklabels([str(x).replace('.00', '').replace('.0', '') for x in Q_labels], fontsize=6)

    ax.grid(linestyle=':', lw=0.2, c='grey', which='minor')
    ax.grid(linestyle='-', lw=0.4, c='darkgrey')

    ax.set_xlabel('Druckhöhengefälle J (mm/m)')
    ax.set_ylabel('Durchfluss Q (l/s)')

    ax.set_ylim([min(Q), max(Q)])
    ax.set_xlim([min(J), max(J)])

    ax.tick_params(direction='out', bottom=True, top=True, left=True, right=True, labelbottom=True, labeltop=True,
                   labelleft=True, labelright=True, which='both')

    ax.text(0.15, 11000, 'k = {:0.01f} mm'.format(k), fontsize=22, fontstretch='ultra-condensed', bbox=bbox(5))
    ax.text(340, 1.7, 'v (m/s)', fontsize=12, rotation=-60, bbox=bbox(2))
    ax.text(300, 0.6, 'd (m)', fontsize=12, rotation=30, bbox=bbox(2))

    # _________________________________________________________________________________________________________________
    # figure post processing
    fig.set_size_inches(h=29.7 / 2.54, w=21 / 2.54)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    fig = nomogram()
    k = 0.1  # mm
    fig.savefig('Nomogramm k_{:0.1f}mm'.format(k).replace('.', '') + '.pdf')
    plt.close(fig)

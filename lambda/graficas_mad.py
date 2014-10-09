#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
graficas_mad.py
------------
Realiza las gráficas del grado de adecuación marginal


"""

__author__ = 'juliowaissman'

import numpy as np
import matplotlib.pyplot as plt

from lamda_general import mad


def grafica_mad():
    """
    Realiza la gráfica de los rho para diferentes x xomo lineas


    """
    xi = np.linspace(0, 1, 50)
    x = np.c_[xi, xi, xi, xi, xi]
    rho = np.array([[.1, .3, .5, .7, .9]])

    m = mad(rho, x)[0]

    plt.plot(xi, m[:, 0], 'k:', label=r'$\rho$ = 0.1')
    plt.hold('on')
    plt.plot(xi, m[:, 1], 'k-.', label=r'$\rho$ = 0.3')
    plt.plot(xi, m[:, 2], 'k-', label=r'$\rho$ = 0.5')
    plt.plot(xi, m[:, 3], 'k.', label=r'$\rho$ = 0.7')
    plt.plot(xi, m[:, 4], 'k--', label=r'$\rho$ = 0.9')
    plt.axis([0, 1, 0, 1])

    plt.title(u'El grado de adecuación marginal respecto a ' + r'$\rho$')
    plt.xlabel('descriptor')
    plt.ylabel(u'grado de adecuación marginal')

    plt.legend(loc=9)
    plt.show()


if __name__ == '__main__':
    grafica_mad()
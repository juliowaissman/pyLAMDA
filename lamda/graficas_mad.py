#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Julio Waissman Vilanova'

import numpy as np
import matplotlib.pyplot as plt

from lamda import Lamda


def grafica_mad():
    """
    Realiza la gráfica de los rho para diferentes x como lineas

    Ejemplo

    >>> grafica_mad()

    """
    xi = np.linspace(0, 1, 50)
    x = np.c_[xi, xi, xi, xi, xi]
    rho = np.array([[.1, .3, .5, .7, .9]])

    lamda = Lamda()
    lamda.set_rho(np.array([[.1, .3, .5, .7, .9]]))
    m = lamda.mad(x)[0]

    plt.plot(xi, m[:, 0], label=r'$\rho$ = 0.1', linewidth= 1.8)
    plt.hold('on')
    plt.plot(xi, m[:, 1], label=r'$\rho$ = 0.3', linewidth= 1.8)
    plt.plot(xi, m[:, 2], label=r'$\rho$ = 0.5', linewidth= 1.8)
    plt.plot(xi, m[:, 3], label=r'$\rho$ = 0.7', linewidth= 1.8)
    plt.plot(xi, m[:, 4], label=r'$\rho$ = 0.9', linewidth= 1.8)
    plt.axis([0, 1, 0, 1])

    # plt.plot(xi, m[:, 0], 'k:', label=r'$\rho$ = 0.1')
    # plt.hold('on')
    # plt.plot(xi, m[:, 1], 'k-.', label=r'$\rho$ = 0.3')
    # plt.plot(xi, m[:, 2], 'k-', label=r'$\rho$ = 0.5')
    # plt.plot(xi, m[:, 3], 'k.', label=r'$\rho$ = 0.7')
    # plt.plot(xi, m[:, 4], 'k--', label=r'$\rho$ = 0.9')
    # plt.axis([0, 1, 0, 1])

    #MADplt.title(u'MAD respect to ' + r'$\rho$')
    plt.xlabel(r'$x_i$', fontsize=18)
    plt.ylabel(r'$m_{i,k}$', fontsize=18)

    plt.legend(loc=9)
    plt.show()


if __name__ == '__main__':
    grafica_mad()

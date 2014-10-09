#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lamda_general.py
------------

Las funciones generales de LAMDA

"""


import numpy as np


def mad(rho, x):
    """
    Calcula el grado de adecuación marginal

    :param rho: Un ndarray de K \times n, donde n es el numero de descriptores
                y K es el numero de clases. Las entradas rho_{i,j} \in (0, 1) se eliminan los valores
                0 y 1 para evitar problemas de continuidad.
    :param x: Un ndarray de T \times n, donde T es el numero de ejemplos y n el de descriptores.
              Las entradas x_{ij} \in [0, 1] son pertenencias a etiquetas.

    :return: [M1, M2, ..., MK] k matrices de tamaño de x con los grados de adecuación marginales para
             cada clase.

    """
    mads = []

    for i in range(rho.shape[0]):
        mads.append( np.power(rho[i, :], x) * np.power(1 - rho[i, :], 1 - x))
    return mads


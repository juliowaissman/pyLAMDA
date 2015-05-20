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

def gad(mads, operador):
    """
    Calcula el grado de adequación global para todas las clases

    :param mads: lista de k matrics [M1, ..., Mk] de tamaño similar a x (datos) con los
                 grados de adequación marginal de cada dato y cada descriptor, tal como
                 se calculan con la función mads 
    
    :param operador: función tal que recibe un ndarray de dimensión n, m (con n objetos y 
                     m descriptores) y regrese un ndarray vector columna tal que en la
                     posición i, aplique el operador de agregación seleccionad a los
                     datos del i´-ésimo renglon.
    
    :return: ndarray de dimensión n, k  con el grado de adecuación marginalde cada clase 
             en cada dato, utilizando el operador de agregación seleccionado.
    
    """
    # TODO:Hacer más clara la documentación de la función 
    gads = np.zeros((mads[0].shape[0], len(mads)))
    for (clase, mad) in enumerate(mads):
        gads[:, clase] = operador(mad)
    return gads
        
# TODO: Hacer el sistema en forma de clase con métodos (Lambda)
# TODO: Hacer varias funciones de operadores de agregación para poder aplicar en LAMDA




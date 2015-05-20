#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lamda_general.py
------------

Las funciones generales de LAMDA

"""


import numpy as np


class Lamda(object):
    """
    Clase contenedora de el método LAMDA con los módulos
    básicos, así como contenedor de los parámetros que definen
    el método LAMDA, tal como se definió por Jsep AGUILAR-MARTIN
    y Ramón LOPEZ de MANTRAS en su planteamiento original.

    En este sistema procuramos mantenernos alejados de las 
    modificaciones que se le hicieron al método que pierde sus
    propiedades naturales, tales como la de manejar para la
    toma de desición una escala bipolar uniescala (la cual en
    la versión inicial se considera simétrica).

    El sistema de reconocimiento consta de dos módulos principales:

    1. El de calculo de adecuación marginal (MAD), el cual se realiza por
       paámetro y por clase.

    2. El calculo de grado de adecuación global (GAD), el cual se realiza
       por clase (independientemente de las otras clases).

    El sistema permite el uso de diversos operadores de agregación
    para el calculo del GAD, siempre y cuando cumplan con los requisitos
    necesarios. Por otra parte el MAD lo vamos a mantener fijo, por lo
    menos en una versión inicial.

    """
    
    def __init__(self, operador=None):
        """
        Inicializa la clase Lambda. En principio muy sencillotoooo
        
        """
        self.rho = None
        self.operador = operador

    def set_rho(self, rho):
        """
        Un simple set para agregar rho son hacer porquerías

        """
        self.rho = rho 


    def mad(self, x):
        """
        Calcula el grado de adecuación marginal

        :param x: Un ndarray de T \times n, donde T es el numero de ejemplos y n el de descriptores.
                  Las entradas x_{ij} \in [0, 1] son pertenencias a etiquetas. Para que se pueda
                  realizar la operación, es necesario que `x.shape[1] == self.rho.shape[1]`

        :return: [M1, M2, ..., MK] k matrices de tamaño de x con los grados de adecuación marginales para
                 cada clase.

        """
        mads = []

        for i in range(self.rho.shape[0]):
            mads.append( np.power(self.rho[i, :], x) * np.power(1 - self.rho[i, :], 1 - x))
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
            
# TODO: Hacer varias funciones de operadores de agregación para poder aplicar en LAMDA

def mixto_maxmin(mads, alpha):
    """
    Operador de agregación mixto utilizando min como t-norma y max como t-conorma.

    # TODO: Acabar la documentación 

    """
    # TODO: acabar con la función prometida.



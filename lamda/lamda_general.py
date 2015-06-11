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
    
    def __init__(self, operador, descriptores=None, conceptos=None):
        """
        Inicializa la clase Lambda. En principio muy sencillotoooo

        :param operador: función tal que recibe un ndarray de dimensión n, m (con n objetos y
                         m descriptores) y regrese un ndarray vector columna tal que en la
                         posición i, aplique el operador de agregación seleccionad a los
                         datos del i´-ésimo renglon. Se puede generar con el decorador
                         @vectorize

        :param descriptores: Entero con el número de descriptores del problema. Si None
                             entonces no se conocen a priori el número de descriptores

        :param conceptos: Entero con el número de clases iniciales si se conocen, Si no, se
                          asume que no se conocen a priori

        """
        self.d = d = descriptores
        self.k = k = conceptos
        self.rho = (np.zeros((k, d)) 
                    if d is not None and k is not None else None)
        self.operador = operador

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

    def gad(self, mads):
        """
        Calcula el grado de adequación global para todas las clases

        :param mads: lista de k matrics [M1, ..., Mk] de tamaño n \times d con los
                     grados de adequación marginal de cada dato y cada descriptor en cada clase,
                     tal como se calculan con la función mads
        
        :return: ndarray de dimensión n, k  con el grado de adecuación marginalde cada clase
                 en cada dato, utilizando el operador de agregación.
        
        """
        gads = np.zeros((mads[0].shape[0], len(mads)))
        for (clase, mad) in enumerate(mads):
            gads[:, clase] = self.operador(mad)
        return gads
            
    def aprendizaje_supervisado(self, x, y):
        """
        Aprendizaje supervisado de la forma tradicional como se conoce en LAMDA
        utilizando simplemente las medias para establecer los valores de Rho.

        En este caso no guarda los valores anteriores, y simplemente vuelve a
        hacer a la matriz rho desde 0. Las clases (sus equivalencias en dado caso)
        las guarda

        """


def vectoriza(oa):
    """
    Decorador para utilizar un operador de agregación dentro de LAMDA

    param oa: Un operador de agregación que funciona sobre un ndarray de
              una dimensión y regresa un valor numérico. El primer parámetro
              de la función oa debe de ser un ndarray de una dimensión, y
              los restantes parámetro que definan el operador

    return Un operador modificado

    Ejemplo:

    @vectoriza
    def luk_tn(x):
        "T-norma de luckasiewicz"
        return max(sum(x) - x.size + 1, 0)

    y se puede probar con
    luk(np.array([[.5, .5, .5],[0, .99, .99],[.9, .9, .9]]))

    """
    def _oa(*args):
        if type(args[0]) != np.ndarray or args[0].ndim > 2:
            raise TypeError("Debe de ser un ndarray de 1 o 2 dimensiones")
        if args[0].ndim == 1:
            return oa(*args)
        y = np.zeros(args[0].shape[0])
        for i in range(args[0].shape[0]):
            y[i] = oa(args[0][i,:]) if len(args) < 2 else oa(args[0][i,:], *args[1:])
        return y
    return _oa


@vectoriza
def op_compensacion(x, tnorma, tconorma, alpha):
    """
    Operador de agregación mixto

    @param x: Un ndarray de una dimensión
    @param tnorma: Una función que recibe un vector y devuelve un número
    @param tconorma: Una función que recibe un vector y devuelve un número
    @alpha: un valor entre 0 y 1

    @return el operador

    ejemplo:
    
    om_7 = lambda x: op_compensacion(x, np.min, np.max, 0.7)

    """
    if 0 > alpha or alpha > 1:
        raise ValueError("alpha entre 0 y 1")
    return alpha * tnorma(x) + (1 - alpha) * tconorma(x)


@vectoriza
def triple_prod(x):
    """
    Operador triple producto tal como lo define Yager en el artículo de
    operadores de agregación completamente reforzados.

    param x: Un ndarray de 1 dimensión

    return un número

    """
    return np.prod(x) / (np.prod(x) + np.prod(1 - x))

if __name__ == "__main__":

    a = np.array([[0, .9, .5],[1, .9, .5],[.1, .1, .1], [.5, .5, .5]])
    print "Matriz para probar los oa"
    print a

    @vectoriza
    def luck_tn(x):
        "T-norma de luckasiewicz"
        return max(sum(x) - x.size + 1, 0)

    print "Luckasiewicz"
    print luck_tn(a)

    print "Triple producto"
    print triple_prod(a)

    om_9 = lambda x: op_compensacion(x, np.min, np.max, 0.9)
    print "O. compensación min/max con exigencia 0.9"
    print om_9(a)

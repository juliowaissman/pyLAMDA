#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from functools import wraps


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

   Para inicializar la clase Lambda, en principio muy sencillito

    :operador: Función tal que recibe un ndarray de dimensión n, m (con n objetos y
                     m descriptores) y regrese un ndarray vector columna tal que en la
                     posición i, aplique el operador de agregación seleccionad a los
                     datos del i-ésimo renglon. Se puede generar con el decorador
                     `@vectorize`.

    :descriptores: Entero con el número de descriptores del problema. Si `None`
                   entonces no se conocen a priori el número de descriptores

    :conceptos: Lista con el nombre de los conceptos (puden ser numeros enteros tambien),
                si `None`, se asume que no se conocen a priori.


    Este ejemplo se puede probar:

    >>> lamda = Lamda(lambda x: tnorma(x, np.min)) #  Un objeto Lamda con el OA del mínimo
    >>> x = np.random.random((10, 3))
    >>> y = np.array([1, 3, 1, 3, 3, 3, 1, 1, 1, 3])
    >>> lamda.aprendizaje_supervisado(x, y)  #  Aprende con los datos generados en x y y
    >>> (yest, gads) = lamda.reconoce(x, gads=True)
    >>> print "rho = "
    >>> print "data =", x
    >>> print "Clases = ", y
    >>> print "Estimados", yest
    >>> print "Adecuaciones", gads


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

        :param conceptos: Lista con el nombre de los conceptos (puden ser numeros enteros tambien),
                          Si None, se asume que no se conocen a priori.

        """
        self.d = d = descriptores
        self.k = k = conceptos
        self.rho = (0.5 * np.ones((len(k), d))
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
        las guarda.

        :param x: Un ndarray de shape (n, d) donde n es el número de objetos y
                  d es el número de descriptores.

        :param y: Un ndarray de shape (d) con los d valores de salida de los datos.
                  si self.k ya existe, los elementos de otras clases nuevas no se
                  considerarán y las clases sin datos se ponen todos los rhos a
                  0.5. Si self.k es None, se genera a partir de los datos las
                  clases. En todo caso, se inicializan los rhos a 0

        """
        if self.d is not None and self.d != x.shape[1]:
            raise ValueError("Los descriptores no concuerdan con la dimensión de los datos")
        y = y.astype(int)
        if self.d is None:
            self.d = x.shape[1]
        if self.k is None:
            self.k = list(np.unique(y))
        self.rho = 0.5 * np.ones((len(self.k), self.d))
        for (i, clase) in enumerate(self.k):
            if clase in y:
                self.rho[i, :] = x[y == clase, :].mean(axis=0)
        return True

    def reconoce(self, x, criterio='max', gads=False):
        """
        Realiza el reconocimiento de un conjunto de variables por reconocer.

        :param x: Un ndarray de shape (n, d) donde n es el número de objetos y
                  d es el número de descriptores.

        :param criterio: Si 'max' entonces asigna a la clase con mayor GAD

        :param gads: Booleano, si True, devuelve una matriz de grados de adequación
                     de dimensión (n, len(k))

        :return: Un ndarray de una dimensión con las clases asignadas a cada objeto
                 y si el parámetro gads es True, una tupla con la asignación, y con las
                 adecuaciones globales.

        """
        if x.shape[1] != self.d:
            raise ValueError("La entrada no concuerda en dimensiones con los descriptores")
        globales = self.gad(self.mad(x))
        asigna = np.vectorize(lambda ind: self.k[ind])
        return (asigna(globales.argmax(axis=1)), globales) if gads else asigna(globales.argmax(axis=1))


def vectoriza(oa):
    """
    Decorador para utilizar un operador de agregación dentro de LAMDA

    :param oa: Un operador de agregación que funciona sobre un ndarray de
              una dimensión y regresa un valor numérico. El primer parámetro
              de la función oa debe de ser un ndarray de una dimensión, y
              los restantes parámetro que definan el operador

    :return Un operador modificado

    Ejemplo:

    >>> @vectoriza
    >>> def luk_tn(x):
    >>>     "T-norma de luckasiewicz"
    >>>     return max(sum(x) - x.size + 1, 0)

    y se puede probar con

    >>> luk(np.array([[.5, .5, .5],[0, .99, .99],[.9, .9, .9]]))

    """
    @wraps(oa)
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
def tnorma(x, fun):
    """
    Una t-norma en forma genérica para funcionar en la clase Lamda como operador de agregación

    :param x: Un ndarray de shape (n, d) donde n es el número de objetos y
              d es el número de descriptores, o un ndarray de shape (n).
    :param fun: Una función que recibe un ndarray de una dimensión y regresa un numero. Se asume que la función
                va a ser una T-norma, pero no se verifica.
    :return: Un ndarray de dimensión (n) con la aplicación de la T-norma a cada caso, o un número en su caso

    Ejemplo:

    >>> min_tnorma = lambda x: tnorma(x, np.min)
    >>> min_tnorma(np.array([[0, 0.9, 0.9], [0.5, 0.5, 0.5]]))

    """
    return fun(x)

@vectoriza
def op_compensacion(x, tnorma, tconorma, alpha):
    """
    Operador de agregación mixto

    :param x: Un ndarray de shape (n, d) donde n es el número de objetos y
              d es el número de descriptores, o un ndarray de shape (n).
    :param tnorma: Una función que recibe un vector y devuelve un número
    :param tconorma: Una función que recibe un vector y devuelve un número
    :param alpha: un valor entre 0 y 1

    :return Un ndarray de dimensión (n) con la aplicación de la T-norma a cada caso, o un número en su caso

    Ejemplo:
    
    >>> om_9 = lambda x: op_compensacion(x, np.min, np.max, 0.9)
    >>> a = np.array([[0, .9, .5],[1, .9, .5],[.1, .1, .1], [.5, .5, .5]])
    >>> om_9(a)

    """
    if 0 > alpha or alpha > 1:
        raise ValueError("alpha entre 0 y 1")
    return alpha * tnorma(x) + (1 - alpha) * tconorma(x)


@vectoriza
def triple_prod(x):
    """
    Operador triple producto tal como lo define Yager en el artículo de
    operadores de agregación completamente reforzados.

    :param x: Un ndarray de shape (n, d) donde n es el número de objetos y
              d es el número de descriptores, o un ndarray de shape (n).

    :return Un ndarray de dimensión (n) con la aplicación de la T-norma a cada caso, o un número en su caso

    Ejemplo:

    >>> a = np.array([[0, .9, .5],[1, .9, .5],[.1, .1, .1], [.5, .5, .5]])
    >>> triple_prod(a)

    """
    return np.prod(x) / (np.prod(x) + np.prod(1 - x))

if __name__ == "__main__":

    print "El unittest de los que no sabemos hacerlas todavía"

    print "Probando los operadores de agregación"
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

    print "Probando generar un objeto tipo Lamda y aprendizaje básico"

    print "Un objeto Lamda con el mínimo"
    lamda = Lamda(lambda x: tnorma(x, np.min))
    print "rho = "
    print lamda.rho

    print "Clases ="
    print lamda.d

    x = np.random.random((10, 3))
    y = np.array([1, 3, 1, 3, 3, 3, 1, 1, 1, 3])
    print "data ="
    print x
    print "Clases = "
    print y

    lamda.aprendizaje_supervisado(x, y)
    print "rho = "
    print lamda.rho

    print "Clases ="
    print lamda.k

    (yest, gads) = lamda.reconoce(x, gads=True)
    print "Estimados"
    print yest
    print "Adecuaciones"
    print gads


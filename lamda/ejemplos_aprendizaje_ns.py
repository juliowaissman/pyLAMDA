#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ejemplos de aprendizaje no supervisado para probar poco a poco
el funcionamiento y poder probar con nuevas funcionalidades.


"""

__author__ = 'juliowaissman'

import lamda
import numpy as np
import matplotlib.pyplot as pl


def prueba_umbral():
    """
    Ejemplo de como se obtiene el umbral para el aprendizaje no supervisado
    utilizando tres objetos lamda.Lamda, uno con un operador min, otro con un
    operador prod, otro con un operador triple_prod, con 2, 3, 5 y 10 ejemplos

    :return: True si pasa la prueba, si no truena, al tener puros asserts

    """
    for atributos in [2, 3, 5, 10]:
        lm = lamda.Lamda(lambda x: lamda.tnorma(x, np.min))
        assert float(lm.gad([0.5 * np.ones((1, atributos))])) == 0.5

        lm = lamda.Lamda(lambda x: lamda.tnorma(x, np.prod))
        assert float(lm.gad([0.5 * np.ones((1, atributos))])) == 0.5 ** atributos

        lm = lamda.Lamda(lamda.triple_prod)
        assert float(lm.gad([0.5 * np.ones((1, atributos))])) == 0.5

    return True


# TODO: Banco de pruebas para el aprendizaje no supervisado, caso sencillo

# TODO: Ejemplo a mano de como funciona el aprendizaje no supervisado en linea


if __name__ == '__main__':

    if prueba_umbral():
        print "Se encuentra el umbral de manera correcta"

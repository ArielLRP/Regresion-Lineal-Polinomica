# -*- coding: utf-8 -*-
"""
Created on Tue May 31 17:16:26 2022

@author: ariel
"""

# =============================================================================
# Regresion Polinomica
# =============================================================================

#Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar el dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values 

#Veamos como se distribuyen los puntos

dataset.plot.scatter(x = 'Level', y = 'Salary')
plt.scatter(dataset.x, dataset.y)

#Ajustar la regresion lineal con el dataset

from sklearn.linear_model import LinearRegression

Lin_reg = LinearRegression()
Lin_reg.fit(X, y)

#Ajustar la regresion polinomica con el dataset

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

Lin_reg_2 = LinearRegression()
Lin_reg_2.fit(X_poly, y)

#Visualizacion de los resultados del Modelo Lineal

plt.scatter(X, y, color = 'red')
plt.plot(X, Lin_reg.predict(X), color = 'blue')
plt.title('Modelo de Regresion Lineal')
plt.xlabel('Posicion del empleado')
plt.ylabel('Sueldo (en dolares)')
plt.show()
#Visualizacion de los resultados del Modelo Polinomico

# plt.scatter(X, y, color = 'red')
# plt.plot(X, Lin_reg_2.predict(X_poly), color = 'blue')
# plt.title('Modelo de Regresion Lineal Polinomica')
# plt.xlabel('Posicion del empleado')
# plt.ylabel('Sueldo (en dolares)')
# plt.show()

#Suavizar grafico
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

#Grafico
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, Lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Modelo de Regresion Lineal Polinomica')
plt.xlabel('Posicion del empleado')
plt.ylabel('Sueldo (en dolares)')
plt.show()

#Prediccion de nuestro modelo 

Lin_reg.predict([[6.5]])

Lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
"""
El objetivo de este trabajo consiste en:
    - Realizar el procesamiento de un set de datos de la central térmica de Argentina Central Costanera S.A. mediante el uso del lenguaje programación Python y las bibliotecas Pandas y Numpy.
    - Obetener información necesaria del dataset con Pandas y desarrollar visualizaciones básicas de los datos mediante la librería MatPlotlib.
    - Ajustar los datos de origen a una serie de funciones de densidad de probabilidad sugeridas, mediante el uso de la biblioteca Fitter.
    - Con la obtención de la(s) fdp(s) sugeridas, simular un array de datos y verificar por medio de gráficos o un nuevo ajuste, que los datos generados son de características similares a los de orígen.

Procedimiento:    
Para el paso a paso de la realización de las fdps se mostrará el ejemplo de operaciones sobre 1 fdp, mientras que los pasos que sean fundamentales para la generación de las mismas
se realizara con todas.
"""

# 
"""
## 1. Importación de bibliotecas
En este primer paso importamos todas las bibliotecas necesarias para realizar nuestro procesamiento de datos. Numpy y Pandas para el manejo de datos y Matplotlib para gráficos.
Generalmente es buena práctica utilizar alias para los nombres de estas bibliotecas (np, pd y plt) cuando se las importa.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from fitter import Fitter
import os
import pandas as pd

"""
## 2. Carga de datos
Debido a que nuestro dataset a procesar es un archivo csv, utilizaremos la función *read_csv* de la biblioteca Pandas.
Los detalles del uso de esta función se encuentran en el siguiente link:
(https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)
"""

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  
CSV_DIR = os.path.join(BASE_DIR, "public")

csv_Autodescarga_BESS = pd.read_csv(os.path.join(CSV_DIR, "Autodescarga_BESS.csv"))
csv_Costo_Falla = pd.read_csv(os.path.join(CSV_DIR, "Costo_Falla.csv"))
csv_Demanda_Primer_Semestre = pd.read_csv(os.path.join(CSV_DIR, "Demanda_Primer_Semestre.csv"))
csv_Demanda_Segundo_Semestre = pd.read_csv(os.path.join(CSV_DIR, "Demanda_Segundo_Semestre.csv"))
csv_Generacion_CC1 = pd.read_csv(os.path.join(CSV_DIR, "Generacion_CC1.csv"))
csv_Generacion_CC2 = pd.read_csv(os.path.join(CSV_DIR, "Generacion_CC2.csv"))
csv_Generacion_TV = pd.read_csv(os.path.join(CSV_DIR, "Generacion_TV.csv"))
csv_Potencia_Perdida = pd.read_csv(os.path.join(CSV_DIR, "Potencia_Perdida.csv"))

# Con la instrucción *type* podemos ver que esta última varible se trata de un *DataFrame* de Pandas.
type(csv_Autodescarga_BESS)

# Podemos pensar a un DataFrame como una típica tabla que alguna vez utilizamos en un motor de bases de datos o simplemente en Excel.
"""
## 3. Analisis Exploratorio de datos
- Lo primero que podemos realizar con nuestro DataFrame es ver su "dimensionalidad" (Cantidad de filas por columnas).
Para ello utilizamos el atributo (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shape.html) de Pandas.

- Podemos ver el contenido de los primeros registros del DataFrame mediante la función head()
Podemos pasar como parámetro la cantidad de primeros x registros a ver. En el siguiente ejemplo, los 10 primeros.

- Oberservamos los tipos de datos de cada una de las columnas con el atributo dtypes.

- Para este trabajo, nos interesa la columna "Autodescarga_Porcentaje", vemos que los mismos son de tipo float

- Una forma de verificar si el DataFrame contiene datos nulos es mediante la función `isnull()`. Si agregarmos `sum()` obtenemos el total de nulos por columnas.
En nuestro dataset observamos que no tenemos datos nulos.

- Guardaremos la cantidad de filas que tiene el archivo para utilizar más adelante en el Array
"""

## - Dimensionalidad
csv_Autodescarga_BESS.shape

## - Dimensionalidad
csv_Autodescarga_BESS.head(10)

## - Tipo de datos 
csv_Autodescarga_BESS.dtypes

## - Inspección de datos nulos
csv_Autodescarga_BESS.isnull().sum()

## - Se guarda la cantidad de filas que tiene el archivo para utilizar más adelante en el array
num_filas_Autodescarga_BESS = csv_Autodescarga_BESS.shape[0]
num_filas_Costo_Falla = csv_Costo_Falla.shape[0]
num_filas_Demanda_Segundo_Semestre = csv_Demanda_Segundo_Semestre.shape[0]
num_filas_Generacion_CC1 = csv_Generacion_CC1.shape[0]
num_filas_Generacion_CC2 = csv_Generacion_CC2.shape[0]
num_filas_Generacion_TV = csv_Generacion_TV.shape[0]
num_filas_Potencia_Perdida = csv_Potencia_Perdida.shape[0]

## Para acceder a los datos de una columna específica de un DataFrame podemos utilizar el operador punto (.), es decir ponemos el nombre del DF, más un . y el nombre de la columna. De esta manera:
csv_Autodescarga_BESS.Autodescarga_Porcentaje

"""
## 4. Visualización de Datos
Es buena práctica recurrir a una visualización para obtener un panorama de la distribución de nuestros datos.
Recurrimos a la biblioteca Matplotlib para graficar un histograma o gráfico de frecuencia.
Para construir este gráfico utilizamos la función hist de Matplotlib y pasamos como parámetros la columna con los valores que queremos contar (tiempo_uso e intervalo_arribos_min) 
y la cantidad de rangos o bins que consideramos para nuestros datos.
"""

# Histograma ejemplo de Autodescarga
plt.title("Histograma Autodescarga_BESS")
plt.xlabel("Autodescarga Porcentaje (%)")
plt.ylabel("Frecuencia")
plt.hist(csv_Autodescarga_BESS.Autodescarga_Porcentaje, bins=100)
plt.show()

"""
5. Ajuste de Datos
Para poder ajustar nuestros datos a una fdp conocida es necesario que previamente instalemos en nuestro entorno Colab el paquete de la biblioteca Fitter.
La instalación se realiza fácilmente mediante la instrucción pip install.
Una vez instalado el paquete, se nos pedirá reiniciar el entorno de Colab.

La función fit() de Fitter ajusta los datos recorriendo más de 80 distribuciones disponibles de la biblioteca scipy que previamente importamos. 
Podríamos indicarle en forma de lista, distribuciones con las cuales queremos que realice la prueba de ajuste, si es que conocemos de antemano que nuestros datos ajustará a alguna de ellas. Por ejemplo así:
fit(datos, distributions=['gamma','rayleigh','uniform'])
Tener en cuenta que si no pasamos ningún parámetro, la función fit() escaneará las 80 funciones de scipy y demorará varios minutos en el proceso.

La función sumary():
summary(Nbest=5, lw=2, plot=True, method='sumsquare_error', clf=True)
grafica la distribución de las N mejores distribuciones.

Fitter me devuelve las mejores distribuciones de ajuste de acuerdo a distintos criterios (sumsquare_error, aic, bic, kl_div, ks_statistic,ks_pvalue). 
Podemos por ejemplo obtener la mejor distribución, considerando el criterio de la suma residual de cuadrados.

En este último ejemplo nos devuelve la mejor distribución para la autodescarga.
No está demás que con cada distribución sugerida, consultemos la documentación de la misma en Scipy.
Investigar en la documentación de Scipy los parámetros específicos de cada distribución con la que elijamos trabajar.
"""
# Es necesario importar Fitter: pip install fitter

# Cargamos los datos de la columna de nuestro DataFrame que queremos ajustar en una variable que llamamos:
# f_autodescarga, por ejemplo.

csv_Autodescarga_BESS['Autodescarga_Porcentaje'] = csv_Autodescarga_BESS['Autodescarga_Porcentaje'].fillna(0)
csv_Costo_Falla['Costo_Falla_USD'] = csv_Costo_Falla['Costo_Falla_USD'].fillna(0)
csv_Demanda_Primer_Semestre['Demanda_MWh'] = csv_Demanda_Primer_Semestre['Demanda_MWh'].fillna(0)
csv_Demanda_Segundo_Semestre['Demanda_MWh'] = csv_Demanda_Segundo_Semestre['Demanda_MWh'].fillna(0)
csv_Generacion_CC1['Generacion_MW'] = csv_Generacion_CC1['Generacion_MW'].fillna(0)
csv_Generacion_CC2['Generacion_MW'] = csv_Generacion_CC2['Generacion_MW'].fillna(0)
csv_Generacion_TV['Generacion_MW'] = csv_Generacion_TV['Generacion_MW'].fillna(0)
csv_Potencia_Perdida['PerdidaCC1_Pct'] = csv_Potencia_Perdida['PerdidaCC1_Pct'].fillna(0)
csv_Potencia_Perdida['PerdidaCC2_Pct'] = csv_Potencia_Perdida['PerdidaCC2_Pct'].fillna(0)

# Cargar datos de 'Autodescarga_Porcentaje' en variable 'f_autodescarga'
f_autodescarga = Fitter(csv_Autodescarga_BESS.Autodescarga_Porcentaje)
f_autodescarga.fit(n_jobs=1)
f_autodescarga.summary(10)

f_costoFalla = Fitter(csv_Costo_Falla.Costo_Falla_USD)
f_costoFalla.fit(n_jobs=1)
f_costoFalla.summary(10)

f_demanda_primer_semestre = Fitter(csv_Demanda_Primer_Semestre.Autodescarga_Porcentaje)
f_demanda_primer_semestre.fit(n_jobs=1)
f_demanda_primer_semestre.summary(10)

f_demanda_segundo_semestre = Fitter(csv_Demanda_Segundo_Semestre.AutodeDemanda_MWhscarga_Porcentaje)
f_demanda_segundo_semestre.fit(n_jobs=1)
f_demanda_segundo_semestre.summary(10)

f_generacion_cc1 = Fitter(csv_Generacion_CC1.Generacion_MW)
f_generacion_cc1.fit(n_jobs=1)
f_generacion_cc1.summary(10)

f_generacion_cc2 = Fitter(csv_Generacion_CC2.Generacion_MW)
f_generacion_cc2.fit(n_jobs=1)
f_generacion_cc2.summary(10)

f_generacion_tv = Fitter(csv_Generacion_TV.Generacion_MW)
f_generacion_tv.fit(n_jobs=1)
f_generacion_tv.summary(10)

f_potencia_perdida1 = Fitter(csv_Potencia_Perdida.PerdidaCC1_Pct)
f_potencia_perdida1.fit(n_jobs=1)
f_potencia_perdida1.summary(10)

f_potencia_perdida2 = Fitter(csv_Potencia_Perdida.PerdidaCC2_Pct)
f_potencia_perdida2.fit(n_jobs=1)
f_potencia_perdida2.summary(10)

# Se obtienen las mejores distribuciones:
best_f_autodescarga = f_autodescarga.get_best(method='sumsquare_error')
best_f_costoFalla = f_costoFalla.get_best(method='sumsquare_error')
best_f_demanda_primer_semestre = f_demanda_primer_semestre.get_best(method='sumsquare_error')
best_f_demanda_segundo_semestre = f_demanda_segundo_semestre.get_best(method='sumsquare_error')
best_f_generacion_cc1 = f_generacion_cc1.get_best(method='sumsquare_error')
best_f_generacion_cc2 = f_generacion_cc2.get_best(method='sumsquare_error')
best_f_generacion_tv = f_generacion_tv.get_best(method='sumsquare_error')
best_f_potencia_perdida1 = f_potencia_perdida1.get_best(method='sumsquare_error')
best_f_potencia_perdida2 = f_potencia_perdida2.get_best(method='sumsquare_error')

"""
Lo que devolverá algo como:
{'nct': {'df': 23.596711533831762,
  'nc': 4.565938634878238,
  'loc': -0.19822327598148462,
  'scale': 27.044843308438672}}
"""

print(best_f_autodescarga)
print(best_f_costoFalla)
print(best_f_demanda_primer_semestre)
print(best_f_demanda_segundo_semestre)
print(best_f_generacion_cc1)
print(best_f_generacion_cc2)
print(best_f_generacion_tv)
print(best_f_potencia_perdida1)
print(best_f_potencia_perdida2)

"""
## 6. Verifcación
La idea es que mediante la biblioteca de funciones estadística scipy.stats generamos un array de datos con esta distribución
con los parámetros que había devuelto get_best()
"""

# f_verif_autodescarga = stats.nct.rvs(df, nc, loc, scale, size=4000, random_state=None)

# # Como mencionamos, el tipo de dato de estas variables creadas es un array de numpy.
# type(f_verif_autodescarga)

# # En el parámetro size le indicamos que el tamaño de este array es de ${num_filas}.
# # De esta manera, pudimos "simular" un set de datos similar a los datos de origen con los cuales inciamos nuestro análisis.

# f_verif_autodescarga

# # Histograma de datos de verificación
# plt.title("Histograma Autodescarga")
# plt.xlabel("Autodescarga Porcentaje (%)")
# plt.ylabel("Frecuencia")
# plt.hist(f_verif_autodescarga, bins=300)
# plt.show()

# # Opcional
# # De la misma manera, si realizamos un buen análisis, aplicando los datos de verificación a Fitter debería ajustar a la función previamente sugerida:
# v_autodescarga = Fitter(f_verif_autodescarga, distributions=['uniform', 'rel_breitwigner','skewcauchy','norminvgauss', 'burr12'])
# v_autodescarga.fit()
# v_autodescarga.summary(6)

"""
7. Conclusión
Observando el array de datos generados con la función de stats.rel_breitwigner y posteriormente observando su distribución en el histograma podemos concluir sobre la similitud con los datos de origen.
Para ser mucho más exhaustivo podemos con este array simulado volver a ajustar los datos con Fitter como hicimos en la última parte del inciso 6, y entre la lista de funciones sugeridas por Fitter, 
vemos que nuevamente se encuentra la función stats.rel_breitwigner.
"""
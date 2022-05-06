"""
Regresión Lineal Univariada
-----------------------------------------------------------------------------------------

En este laboratio se construirá un modelo de regresión lineal univariado.

"""
from re import X
import numpy as np
import pandas as pd


def pregunta_01():
    """
    En este punto se realiza la lectura de conjuntos de datos.
    Complete el código presentado a continuación.
    """
    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv('gm_2008_region.csv', sep=',')
    #df.head()
    #df.info()

    # Asigne la columna "life" a `y` y la columna "fertility" a `X`
    #y = df['life'].map(lambda x: x for x in df)
    y = df['life']
    X = df['fertility']

    # Imprima las dimensiones de `y`
    print(y.shape)

    # Imprima las dimensiones de `X`
    print(X.shape)

    # Transforme `y` a un array de numpy usando reshape
    y_reshaped = y.values.reshape(-1,1)
    #le pones values o to_numpy

    # Trasforme `X` a un array de numpy usando reshape
    X_reshaped = X.values.reshape(-1,1)

    # Imprima las nuevas dimensiones de `y`
    print(y_reshaped.shape)

    # Imprima las nuevas dimensiones de `X`
    print(X_reshaped.shape)


def pregunta_02():
    """
    En este punto se realiza la impresión de algunas estadísticas básicas
    Complete el código presentado a continuación.
    """

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv('gm_2008_region.csv', sep=',')

    # Imprima las dimensiones del DataFrame
    print(df.shape)

    # Imprima la correlación entre las columnas `life` y `fertility` con 4 decimales.
    print(round(df['life'].corr(df['fertility'], method='pearson'),4))

    # Imprima la media de la columna `life` con 4 decimales.
    print(round(df['life'].mean(),4))

    # Imprima el tipo de dato de la columna `fertility`.
    print(df['fertility'].__class__)

    # Imprima la correlación entre las columnas `GDP` y `life` con 4 decimales.
    print(round(df['life'].corr(df['GDP'], method='pearson'),4))


def pregunta_03():
    """
    Entrenamiento del modelo sobre todo el conjunto de datos.
    Complete el código presentado a continuación.
    """

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv('gm_2008_region.csv', sep=',')

    # Asigne a la variable los valores de la columna `fertility`
    X_fertility = np.array(df['fertility'])

    # Asigne a la variable los valores de la columna `life`
    y_life = np.array(df['life'])

    # Importe LinearRegression
    #https://github.com/scikit-learn/scikit-learn/issues/18621
    from sklearn.linear_model import LinearRegression

    # Cree una instancia del modelo de regresión lineal
    reg = LinearRegression()

    # Cree El espacio de predicción. Esto es, use linspace para crear
    # un vector con valores entre el máximo y el mínimo de X_fertility
    prediction_space = np.linspace(
        X_fertility.min(),
        X_fertility.max(),
        num=139
    ).reshape(-1, 1)

    # Entrene el modelo usando X_fertility y y_life
    #reg.fit(X_fertility.values.reshape(-1,1), y_life.values.reshape(-1,1))
    reg.fit(X_fertility.reshape(-1,1), y_life.reshape(-1,1))
    #print("lr.coef_: {}".format(reg.coef_))
    #print("lr.intercept_: {}".format(reg.intercept_))

    # Compute las predicciones para el espacio de predicción
    y_pred = reg.predict(prediction_space)

    # Imprima el R^2 del modelo con 4 decimales
    #print("%.4f" % r2_score(y_life, y_pred=))...diferencia con la función r2_score???
    print("{:.4f}".format(reg.score(X_fertility.reshape(-1,1), y_life.reshape(-1,1))))



def pregunta_04():
    """
    Particionamiento del conjunto de datos usando train_test_split.
    Complete el código presentado a continuación.
    """

    # Importe LinearRegression
    # Importe train_test_split
    # Importe mean_squared_error
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    #from math import sqrt

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv('gm_2008_region.csv', sep=',')

    # Asigne a la variable los valores de la columna `fertility`
    X_fertility = np.array(df['fertility'])

    # Asigne a la variable los valores de la columna `life`
    y_life = np.array(df['life'])

    # Divida los datos de entrenamiento y prueba. La semilla del generador de números
    # aleatorios es 53. El tamaño de la muestra de entrenamiento es del 80%
    (X_train, X_test, y_train, y_test,) = train_test_split(
        X_fertility,
        y_life,
        test_size=0.2,
        random_state=53,
    )

    # Cree una instancia del modelo de regresión lineal
    linearRegression = LinearRegression()

    # Entrene el clasificador usando X_train y y_train
    #https://www.iartificial.net/regresion-lineal-con-ejemplos-en-python/
    linearRegression.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))
    #print("lr.coef_: {}".format(linearRegression.coef_))
    #print("lr.intercept_: {}".format(linearRegression.intercept_))

    # Pronostique y_test usando X_test
    y_pred = linearRegression.predict(X_test.reshape(-1,1))
    #print(y_pred)

    # Compute and print R^2 and RMSE
    #print("R^2: {:.4f}".format(linearRegression.score(X_train.reshape(-1,1), y_train.reshape(-1,1))))
    print("R^2: {:.4f}".format(linearRegression.score(X_test.reshape(-1,1), y_test.reshape(-1,1))))
    #otra forma de imprimir el r2 del modleo ("R^2: %.4f" % r2_score(y_test, y_pred))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error: {:6.4f}".format(rmse))

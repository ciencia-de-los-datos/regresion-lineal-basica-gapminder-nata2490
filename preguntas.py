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
    y_reshaped = np.reshape(y,-1)
    #le pones values o to_numpy

    # Trasforme `X` a un array de numpy usando reshape
    X_reshaped = np.reshape(X, -1)

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
    X_fertility = df['fertility']

    # Asigne a la variable los valores de la columna `life`
    y_life = df['life']

    # Importe LinearRegression
    #se llama scikit-learn en python 3.9??
    #pip install --pre -U scikit-learn 
    #https://github.com/scikit-learn/scikit-learn/issues/18621
    from sklearn.linear_model import LinearRegression

    # Cree una instancia del modelo de regresión lineal
    reg = LinearRegression()

    # Cree El espacio de predicción. Esto es, use linspace para crear
    # un vector con valores entre el máximo y el mínimo de X_fertility
    prediction_space = np.linspace(
        X_fertility.min(),
        X_fertility.max(),
    ).reshape(____, _____)

    # Entrene el modelo usando X_fertility y y_life
    reg.fit(X_fertility, y_life)

    # Compute las predicciones para el espacio de predicción
    y_pred = reg.predict(prediction_space)

    # Imprima el R^2 del modelo con 4 decimales
    print(reg.score(X_fertility, y_life).round(4))


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
    from sklearn.metrics import mean_squared_error

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv('gm_2008_region.csv', sep=',')

    # Asigne a la variable los valores de la columna `fertility`
    X_fertility = df['fertility']

    # Asigne a la variable los valores de la columna `life`
    y_life = df['life']

    # Divida los datos de entrenamiento y prueba. La semilla del generador de números
    # aleatorios es 53. El tamaño de la muestra de entrenamiento es del 80%
    (X_train, X_test, y_train, y_test,) = train_test_split(
        X,
        y,
        test_size=0.8,
        random_state=53,
    )

    # Cree una instancia del modelo de regresión lineal
    linearRegression = Linear_model.linearRegression()

    # Entrene el clasificador usando X_train y y_train
    linearRegression.fit(X_train, y_train)
    #print("lr.coef_: {}".format(linearRegression.coef_))
    #print("lr.intercept_: {}".format(linearRegression.intercept_))

    # Pronostique y_test usando X_test
    y_pred = linearRegression.predict(X_test)
    #print(y_pred)

    # Compute and print R^2 and RMSE
    #print("R^2: {:6.4f}".format(linearRegression.score(X_train, y_train)))
    print("R^2: {:6.4f}".format(linearRegression.score(X_test, y_test)))
    rmse = np.sqrt(____(____, ____))
    print("Root Mean Squared Error: {:6.4f}".format(rmse))

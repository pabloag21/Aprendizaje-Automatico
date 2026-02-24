import numpy as np
import copy
import math
import matplotlib.pyplot as plt


#########################################################################
# Cost function
#
def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities)
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    m = x.shape[0]                  # número de ejemplos
    y_hat = w * x + b               # predicciones del modelo
    errors = y_hat - y              # vector de errores
    total_cost = (errors @ errors) / (2 * m)   # equivalente a 0.5 * np.mean(errors**2)

    return total_cost


#########################################################################
# Gradient function
#
def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Según lo que tengo entendido, w y b son los valores de la recta y = wx +b
    # Entonces lo que tengo que hacer es calcular dw db que indicarán hacia donde
    # orientar la recta, ya que hay que aproximarlos lo maximo posible a 0.
    # Ahora solo me queda averiguar como hacer esto matematicamnete

    # Numero de ejemplos
    m = x.shape[0]

    # Predicciones
    y_hat = w * x + b

    # Errores (predicciones - valores_reales)
    errors = y_hat - y

    # Derivadas
    dj_db = np.sum(errors) / m
    dj_dw = np.sum(errors * x) / m

    return dj_dw, dj_db


#########################################################################
# gradient descent
#
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar) Updated value of parameter of the model after
          running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
    """

    #Creo el historial que sera un array relleno de ceros, e inicializo w y b
    w = w_in
    b = b_in
    J_history = np.zeros(num_iters)

def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression 
    Returns dj_dw, dj_db
    """
    m = x.shape[0]
    # predicción
    y_hat = w * x + b
    # errores
    errors = y_hat - y
    # gradientes (vectorizado)
    dj_db = np.sum(errors) / m
    dj_dw = np.sum(errors * x) / m
    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn w and b.

    Args:
      x, y: (ndarray) datos
      w_in, b_in: (float) valores iniciales
      cost_function: función que devuelve J(w,b)
      gradient_function: función que devuelve (dj_dw, dj_db)
      alpha: learning rate
      num_iters: número de iteraciones

    Returns:
      w, b, J_history
    """
    w = w_in
    b = b_in
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # actualización simultánea
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # guardar coste para seguimiento
        J_history[i] = cost_function(x, y, w, b)

    return w, b, J_history

# Crear gráfico de los datos iniciales
def graficar_datos(x, y):

    plt.figure(figsize=(6,4))
    plt.scatter(x, y, marker='x', c='red')

    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000")
    plt.title("Profits vs. Population per city")

    plt.show()

#Crear grafico de los datos y la recta de regresion
def graficar_recta(x, y, w, b):
    
    plt.figure(figsize=(6,4))
    plt.scatter(x, y, marker='x', c='red', label='Datos')

    x_vals = np.array([x.min(), x.max()])
    y_vals = w * x_vals + b
    plt.plot(x_vals, y_vals, '-', label='Ajuste lineal')
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000")
    plt.title("Profits vs. Population per city (fit)")
    plt.legend()

    plt.show()

def main():
    # Cargar datos
    data = np.loadtxt("C:/Users/pablo/OneDrive - Universidad Complutense de Madrid (UCM)/Uni/3º/2º/AA/workspace/Practicas/Practica_1/data/ex1data1.txt", delimiter=",")
    x = data[:, 0]
    y = data[:, 1]

    graficar_datos(x, y)

    # inicializar parámetros
    w_init = 0.0
    b_init = 0.0
    alpha = 0.01
    num_iters = 1500

    w, b, J_hist = gradient_descent(x, y, w_init, b_init, compute_cost, compute_gradient, alpha, num_iters)

    print(f"Parámetros aprendidos: w = {w:.8f}, b = {b:.8f}")
    print(f"Coste final: {J_hist[-1]:.6f}")

    # Dibujar recta de ajuste
    graficar_recta(x, y, w, b)

    # Ejemplo de predicción: población 35k y 70k -> x = 3.5 y 7.0
    for pop in [3.5, 7.0]:
        pred = w * pop + b
        print(f"Predicción para población={pop*10000:.0f}: {pred*10000:.2f} $ (en unidades del dataset: {pred:.4f})")




if __name__ == "__main__":
    main()
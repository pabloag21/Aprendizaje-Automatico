import numpy as np
import utils as ut
from scipy.io import loadmat


#########################################################################
# one-vs-all
#
def oneVsAll(X, y, n_labels, lambda_):
    """
     Trains n_labels logistic regression classifiers and returns
     each of these classifiers in a matrix all_theta, where the i-th
     row of all_theta corresponds to the classifier for label i.

     Parameters
     ----------
     X : array_like
         The input dataset of shape (m x n). m is the number of
         data points, and n is the number of features. 

     y : array_like
         The data labels. A vector of shape (m, ).

     n_labels : int
         Number of possible labels.

     lambda_ : float
         The logistic regularization parameter.

     Returns
     -------
     all_theta : array_like
         The trained parameters for logistic regression for each class.
         This is a matrix of shape (K x n+1) where K is number of classes
         (ie. `n_labels`) and n is number of features without the bias.
     """

    return all_theta


def predictOneVsAll(all_theta, X):
    """
    Return a vector of predictions for each example in the matrix X. 
    Note that X contains the examples in rows. all_theta is a matrix where
    the i-th row is a trained logistic regression theta vector for the 
    i-th class. You should set p to a vector of values from 0..K-1 
    (e.g., p = [0, 2, 0, 1] predicts classes 0, 2, 0, 1 for 4 examples) .

    Parameters
    ----------
    all_theta : array_like
        The trained parameters for logistic regression for each class.
        This is a matrix of shape (K x n+1) where K is number of classes
        and n is number of features without the bias.

    X : array_like
        Data points to predict their labels. This is a matrix of shape 
        (m x n) where m is number of data points to predict, and n is number 
        of features without the bias term. Note we add the bias term for X in 
        this function. 

    Returns
    -------
    p : array_like
        The predictions for each data point in X. This is a vector of shape (m, ).
    """

    return p


#########################################################################
# NN
#

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(theta1, theta2, X):
    """
    Predict the label of an input given a trained neural network.

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size)

    X : array_like
        The image inputs having shape (number of examples x image dimensions).

    Return 
    ------
    p : array_like
        Predictions vector containing the predicted label for each example.
        It has a length equal to the number of examples.
    """

    # Hay que multiplicar el vector de entrada por theta1 y el resultado por theta2
    # Tiene pinta sencillo

    m = X.shape[0]


    a1 = np.hstack([np.ones((m, 1)), X]) # Ahora ya tiene el bias
    z1 = a1 @ theta1.T

    a2 = sigmoid(z1) # Pasarlo por el sigmoide

    a2 = np.hstack([np.ones((m, 1)), a2])  # De nuevo el BIAS
    z2 = a2 @ theta2.T

    a3 = sigmoid(z2)
    
    p = np.argmax(a3, axis=1)

    return p

def cargar_datos():
    # Cargar Imagenes y Etiquetas
    data = loadmat('C:/Users/pablo/OneDrive - Universidad Complutense de Madrid (UCM)/Uni/3º/2º/AA/workspace/Practicas/Practica_4/data/ex3data1.mat', squeeze_me=True)
    X = data['X']      # Imagenes
    y = data['y']      # Etiquetas

    # Cargar Matriz de Pesos
    weights = loadmat('C:/Users/pablo/OneDrive - Universidad Complutense de Madrid (UCM)/Uni/3º/2º/AA/workspace/Practicas/Practica_4/data/ex3weights.mat', squeeze_me=True)
    theta1 = weights['Theta1'] # pesos capa 1
    theta2 = weights['Theta2'] # pesos capa 2

    return X, y, theta1, theta2 



def main():
    X, y, theta1, theta2 = cargar_datos()

    # Mostrar algunas imágenes
    #ut.displayData(X)

    # Obtener predicciones
    p = predict(theta1, theta2, X)

    # Calcular precisión
    accuracy = np.mean(p == y) * 100
    print(f"Precisión del modelo: {accuracy:.2f}%")

    # Mostrar algunos ejemplos de predicción
    for i in range(10):
        print(f"Ejemplo {i}: Predicción = {p[i]}, Etiqueta real = {y[i]}")
    

if __name__ == "__main__":
    main()
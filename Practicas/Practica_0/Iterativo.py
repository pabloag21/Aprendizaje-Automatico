from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import time

def integrate_mc(fun, a, b, num_points=10000, plot=False):

    eje_x = np.linspace(a, b, num_points)
    eje_y = fun(eje_x)

    max_f = np.max(eje_y)
    min_f = np.min(eje_y)

    if plot:
        plt.figure()
        plt.plot(eje_x, eje_y, '-')

    debajo = 0

    for i in range(num_points):
        x = np.random.uniform(a, b)
        y = np.random.uniform(min_f, max_f)

        if y <= fun(x):
            debajo += 1
            if plot:
                plt.plot(x, y, 'r.', markersize=1)
        else:
            if plot:
                plt.plot(x, y, 'b.', markersize=1)

    area_rect = (b - a) * (max_f - min_f)
    integral = (debajo / num_points) * area_rect

    if plot:
        plt.show(block=False)
        plt.pause(2)

    return integral


def integrate_mc_vec(fun, a, b, num_points=10000, plot=False):

    eje_x = np.linspace(a, b, num_points)
    eje_y = fun(eje_x)

    max_f = np.max(eje_y)
    min_f = np.min(eje_y)

    x = np.random.uniform(a, b, num_points)
    y = np.random.uniform(min_f, max_f, num_points)

    f_random = fun(x)

    debajo_mask = y <= f_random
    debajo = np.sum(debajo_mask)

    area_total = (b - a) * (max_f - min_f)
    integral = area_total * (debajo / num_points)

    if plot:
        plt.figure()
        plt.plot(eje_x, eje_y, '-')

        # Puntos debajo (rojo)
        plt.plot(x[debajo_mask], y[debajo_mask], 'r.', markersize=2)

        # Puntos encima (azul)
        plt.plot(x[~debajo_mask], y[~debajo_mask], 'b.', markersize=2)

        plt.show()

    return integral


def cuadrado(x):
    return x * x

def lineal(x):
    return x

def main():

    print("Integral de x^2 entre 0 y 1\n")
def main():

    print("Integral de x^2 entre 0 y 1\n")

    # Monte Carlo iterativo
    t0 = time.perf_counter()
    resultado_iter = integrate_mc(lineal, 0, 1, num_points=10000, plot=True)
    t1 = time.perf_counter()
    print("Monte Carlo (iterativo):", resultado_iter)
    print("Tiempo iterativo:", t1 - t0, "segundos\n")

    # Monte Carlo vectorizado
    t0 = time.perf_counter()
    resultado_vec = integrate_mc_vec(lineal, 0, 1, num_points=10000, plot=True)
    t1 = time.perf_counter()
    print("Monte Carlo (vectorizado):", resultado_vec)
    print("Tiempo vectorizado:", t1 - t0, "segundos\n")

    # Resultado exacto con scipy
    t0 = time.perf_counter()
    resultado_quad, error = integrate.quad(cuadrado, 0, 1)
    t1 = time.perf_counter()
    print("Scipy quad:", resultado_quad)
    print("Error estimado por scipy:", error)
    print("Tiempo quad:", t1 - t0, "segundos")

    x = input("Pulsa enter para cerrar")


if __name__ == "__main__":
    main()

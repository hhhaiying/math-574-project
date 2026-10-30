import numpy as np

def rk_system(f,x,a,b,n):
    """Runge kutta 4 algorithm with n time steps
    f : function of time and x
    x : initial value
    a : left time endpoint
    b : right time endpoint
    n : number of time steps"""
    t = a
    h = (b - a) / n
    points = []
    time = []
    for k in range(n):
        x_0 = x.copy()
        points.append(x_0)
        time.append(t)
        K1 = f(t,x)
        y_1 = x+ 0.5*h*K1
        K2 = f(t+0.5*h,y_1)
        y_2 = x+ 0.5*h*K2
        K3 = f(t+0.5*h, y_2)
        y_3 = x + h*K3
        K4 = f(t+h, y_3)
        t += h
        x += 1/6 * h*(K1 + 2*K2 + 2*K3 + K4)
    points = np.array(points)
    time = np.array(time)
    return points,time


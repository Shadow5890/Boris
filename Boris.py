import numpy as np
from Fields import *
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, coords, momentum):
        self.coords = coords
        self.momentum = momentum
        self.size = 10000
        self.time = np.linspace(0, 50, self.size)

    def coord(self, a, b):
        d = {1: 'x', 2: 'y', 3: 'z'}
        plt.plot(self.coords[:, a-1], self.coords[:, b-1])
        plt.xlabel(d[a])
        plt.ylabel(d[b])
        plt.show()

    def coord3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.coords[:, 0], self.coords[:, 1], self.coords[:, 2], label='Trajectory in 3D')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def moment(self, b):
        d = {1: '$p_x$', 2: '$p_y$', 3: '$p_z$'}
        plt.plot(self.time, self.momentum[:, b-1])
        plt.xlabel('t')
        plt.ylabel(d[b])
        plt.show()
        
    def intensity(self):  # график интенсивности на питоне
        n = [1, 0, 0]
        omega = np.linspace(0, 5, 1000)
        result = np.zeros(1000)
        dt = self.time[1] - self.time[0]
        for i in range(1, 1000, 1):  #omega.size()
            J = np.zeros(3)
            for k in range(4000 - 1):   
                r = self.coords[k, :]
                r1 = self.coords[k+1, :]
                xi = self.time[k] - np.dot(n, r)
                xi1 = self.time[k+1] - np.dot(n, r1)
                dxi = xi1 - xi  # нашли dxi
                xi_average = (xi + xi1) / 2
                v = self.momentum[k, :]
                v = v / np.sqrt(1 + np.dot(v, v))  # скорость в начале
                v1 = self.momentum[k+1, :]
                v1 = v1 / np.sqrt(1 + np.dot(v1, v1))  # скорость в конце
                
                dv = v1 - v  # разность скоростей
                v_average = (v + v1) / 2  # средняя скорость
                res = np.exp(1j * omega[i] * xi_average) * dt / dxi * (2 * v_average * np.sin(omega[i] * dxi / 2) + 1j * dv * (np.sin(omega[i] * dxi / 2)/(omega[i] * dxi/2) - np.cos(omega[i] * dxi / 2)))
                J = J + res
            temp = np.cross(n, np.cross(n, J))
            result[i] = omega[i] * np.dot(temp, np.conjugate(temp)) /4 /np.pi**2
        plt.plot(omega, result)
        plt.xlabel('$\omega/\omega_L$')
        plt.ylabel('$d\epsilon / d\Omega d\omega |\Theta = \pi / 2$')
        plt.show()
        
    def intcython(self):
        plo = Intensity(self.coords, self.momentum)
        g = plo.intensity(phi = 0.0, theta = np.pi/2.0)
        plt.plot(np.linspace(0, 5, 1000), g)
        plt.show()
    
    def polar(self):
        w = np.arange(0.01, 10.01, 0.01)
        theta_direct = np.arange(0, 2*np.pi, 0.04)
        data_direct = np.zeros((len(theta_direct), len(w)))
        
        plo = Intensity(self.coords, self.momentum)

        for i in range(len(theta_direct)):
            
            data_direct[i] = plo.intensity(phi=0.0, theta=theta_direct[i])
            print(i)
    
        omega, theta = np.meshgrid(w, theta_direct) 

        fig = plt.figure(figsize = (8, 10))
        ax = fig.add_subplot(111, polar='True')
        im = plt.pcolormesh(theta, omega, data_direct, cmap='inferno')
        plt.colorbar(im, orientation='horizontal')
        #plt.savefig("6.svg")
        plt.show()



fiel = Electromagnetic('crossfield', 100, [0, 0, 1])
par = Particle([0, 0, 0], [-100, 0, 0], fiel)
# на вход подаем: начальные координаты, начальные импульсы, электромагнитное 
# поле. 
solver = Borismethod(par, 1)
# Последний параметр - 0 в случае если мы не добавляем силу трения, иначе 1.
a, b = solver.calculate(0)  # а - импульсы частицы, b - траектория частицы
# на данный момент выполнено построение траектории частицы при наличии или
# отсутствии силы трения (по желанию пользователя) в произвольном э-м поле.
# построим соответствующие графики.
# 1 - электрическое поле включено, 0 - выключено

# график интенсивностей, используем cython
plo = Plotter(b, a) 
plo.coord3D()
# plo = Plotter(b, a)
# plo.intensity()
# Инструкция по работе с классом Plotter. coord - функция, которая построит траекторию
# частицы в 2d: на вход подаются 2 цифры. Каждая отвечает своей координате.
# 1 - координата х. 2 - координата у. 3 - координата z. 
# Пример: coord(1, 2) - график с плоскости xy.
# coord3D - строит график траектории в 3D.
# moment - на вход подается цифра - номер координаты импульса в зависимости от времени.
# функция intensity - там ищем интенсивность в угол тета = pi/2. Чистый python
# функция intcython - делает то же, что и функция intensity, но расчеты проведены с 
# помощью интерфейса cython
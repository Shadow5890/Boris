# -*- coding: utf-8 -*-


import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport pi
from libc.math cimport sin
from libc.math cimport cos
from libc.math cimport sqrt
from cython.parallel import prange, parallel

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False) 
cdef double dot(double [:] a, double [:] b) nogil:  # возвращает скалярное произведение
        cdef int i
        cdef double p
        p = 0
        for i in range(3):
            p = p + a[i] * b[i]
        return p
      
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)               
cdef double cross(double [:] a, double [:] b, int j) nogil:  # возвращаем векторное произведение
    if j == 0:
        return a[1] * b[2] - a[2] * b[1]
    if j == 1: 
        return a[2] * b[0] - a[0] * b[2]
    if j == 2: 
        return a[0] * b[1] - a[1] * b[0]
        
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False) 
cdef double norm(double [:] a) nogil:  # вернем норму вектора
    cdef int i
    cdef double nor
    nor = 0
    for i in range(3):
        nor = nor + a[i] * a[i]
    nor = nor**(1/2)
    return nor

cdef class Electromagnetic:  # класс полей
    cdef:
        double ampl  # амплитуда поля
        double t0  # длительность лазерного импульса (при наличии)
        double [:] k_mv  # волновой вектор (при наличии)
        double [:] curE_mv, curH_mv  #вектора полей
        np.ndarray curE, curH, k
        double [:] k_prob_mv  # вектор [0, -1, 0]
        double [:] new_mv  # вектор [1, 0, 0]
        str name
        
        
    def __init__(self, str name, double ampl = 1, list k1 = [0, 0, 0], double t0 = 1):
        self.ampl = ampl  #амплитуда
        self.k = np.array(k1).astype(np.double)  # волновой вектор
        self.k_mv = self.k
        self.t0 = t0
        self.curE = np.zeros(3)
        self.curH = np.zeros(3)
        self.curE_mv = self.curE
        self.curH_mv = self.curH
        self.name = name
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef int crossfield(self, int a):  # а == 1 - эл. поле, а == 2 - магнитное
        with nogil:
            if (a == 1):
                for i in range(3):
                    if (i == 0):  # поле Е по оси х
                        self.curE_mv[i] = self.ampl
                    else:
                        self.curE_mv[i] = 0
            if (a == 2):
                for i in range(3):
                    if (i == 1):  # поле Н по оси у
                        self.curH_mv[i] = self.ampl
                    else:
                        self.curH_mv[i] = 0
        return 0
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False) 
    cdef int wavefield(self, int a, double [:] r, double t):  # поле э-м волны
        k_prob = np.zeros(3)
        self.k_prob_mv = k_prob
        self.k_prob_mv[1] = -1  # хочу сравнить вектор k с вектором [0, -1, 0]
        cdef int i, j, knox
        cdef np.ndarray new
        new = np.zeros(3)
        self.new_mv = new
        knox = 0
        with nogil:
            for i in range(3):
                if self.k_mv[i] != self.k_prob_mv[i]:  # если вектора не совпали хотя бы раз
                    knox = 1
                    break
            if knox == 1:  # в цикле возврат поэлементно
                for j in range(3):
                    self.curE_mv[j] = cross(self.k_mv, self.k_prob_mv, j) * cos(t * norm(self.k_mv) - dot(self.k_mv, r)) # вот так направляем вектор Е
            else:
                self.new_mv[0] = 1
                for j in range(3):
                    self.curE_mv[j] = cross(self.k_mv, self.new_mv, j) * cos(t * norm(self.k_mv) - dot(self.k_mv, r)) # вот так направляем вектор Е
            if (a == 2):
                for j in range(3):
                    self.curH_mv[j] = cross(self.k_mv, self.curE_mv, j) / norm(self.k_mv)
        return 1

    cdef double e_field(self, double [:] r, double t, int number):  # электрическое поле
        if self.name == "crossfield":
            self.crossfield(1)
            if (number == 0):
                return self.curE_mv[0]
            if (number == 1):
                return self.curE_mv[1]
            if (number == 2):
                return self.curE_mv[2]
        if self.name == "wavefield":
            self.wavefield(1, r, t)
            if (number == 0):
                return self.curE_mv[0]
            if (number == 1):
                return self.curE_mv[1]
            if (number == 2):
                return self.curE_mv[2]
            
        
    cdef double h_field(self, double [:] r, double t, int number):
        if self.name == "crossfield":
            self.crossfield(2)
            if (number == 0):
                return self.curH_mv[0]
            if (number == 1):
                return self.curH_mv[1]
            if (number == 2):
                return self.curH_mv[2]
        if self.name == "wavefield":
            self.wavefield(2, r, t)
            if (number == 0):
                return self.curH_mv[0]
            if (number == 1):
                return self.curH_mv[1]
            if (number == 2):
                return self.curH_mv[2]

cdef class Borismethod:
    cdef:
        double [:] time_mv  # разбиение по времени
        double [:, :] impulses_mv  # импульсы
        double [:, :] coord_mv
        double [:] p_minus_mv, p_plus_mv, p_prime_mv, t_mv, s_mv, currE_mv, currH_mv, v_mv
        double [:] force_mv
        double [:] onecoord_mv
        int size, radiation
        Particle part
        np.ndarray time, impulse, coord, p_minus, p_plus, p_prime, t, s, currE, currH, v, force
        np.ndarray onecoord
        
    def __init__(self, Particle particle, int radiation):
        self.size = 100000 
        self.time = np.linspace(0, 600, self.size)
        self.time_mv = self.time
        self.impulse = np.zeros((self.size, 3)) 
        self.impulses_mv = self.impulse
        self.coord = np.zeros((self.size, 3))
        self.coord_mv = self.coord
        self.onecoord = np.zeros(3)
        self.onecoord_mv = self.onecoord
        self.part = particle  # объект класса частицы, содержащий начальные
        self.radiation = radiation
        
        self.p_minus = np.zeros(3)
        self.p_plus = np.zeros(3)
        self.p_prime = np.zeros(3)
        self.t = np.zeros(3)
        self.s = np.zeros(3)
        self.currE = np.zeros(3)
        self.currH = np.zeros(3)
        self.v = np.zeros(3)
        self.force = np.zeros(3)
        
        self.p_minus_mv = self.p_minus
        self.p_plus_mv = self.p_plus
        self.p_prime_mv = self.p_prime
        self.t_mv = self.t
        self.s_mv = self.s
        self.currE_mv = self.currE
        self.currH_mv = self.currH
        self.v_mv = self.v
        self.force_mv = self.force
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False) 
    cpdef calculate(self, int efi):
        cdef int i, j
        cdef double dt
        cdef double temp
        cdef double K
        temp = 0
        dt = self.time_mv[1] - self.time_mv[0]
        for i in range(3):  # забиваем начальные условия
            self.impulses_mv[0][i] = self.part.p_mv[i]
            self.coord_mv[0][i] = self.part.x_mv[i]
        for i in range(1, self.size, 1):
            #сначала получим вектор текущей точки для полей
            for j in range(3):
                self.onecoord_mv[j] = self.coord_mv[i-1][j]
            if (efi == 1):
                for j in range(3):
                    self.currE_mv[j] = self.part.field.e_field(self.onecoord_mv, self.time_mv[i], j)
            if (efi == 0):
                for j in range(3):
                    self.currE_mv[j] = 0
            temp = 0
            for j in range(3):
                self.currH_mv[j] = self.part.field.h_field(self.onecoord_mv, self.time_mv[i], j)  # нашли магнитное поле
            for j in range(3):
                temp = temp + self.impulses_mv[i-1][j]**2  # вычислили p^2 для гаммы
            for j in range(3):
                self.t_mv[j] = self.currH_mv[j] * dt / 2 / (sqrt(1 + temp))  # сделали вектор t
            for j in range(3):
                self.s_mv[j] = 2 * self.t_mv[j] / (1 + dot(self.t_mv, self.t_mv))  # сделали вектор s
            for j in range(3):
                self.p_minus_mv[j] = self.impulses_mv[i-1][j] + self.currE_mv[j] * dt / 2  # сделали p-
            for j in range(3):
                self.p_prime_mv[j] = self.p_minus_mv[j] + cross(self.p_minus_mv, self.t_mv, j)  # сделали p`
            for j in range(3):
                self.p_plus_mv[j] = self.p_minus_mv[j] + cross(self.p_prime_mv, self.s_mv, j)  # сделали p+
            for j in range(3):
                self.impulses_mv[i][j] = self.p_plus_mv[j] + self.currE_mv[j] * dt / 2
            # на данный момент готовы импульсы без учета радиационного трения
            temp = 0
            for j in range(3):
                temp = temp + self.impulses_mv[i-1][j]**2  # обновили гамму для координат
            # на данный момент вычислено всё, без учета силы радиационного трения. вычислим.
            for j in range(3):
                self.v_mv[j] = (self.impulses_mv[i][j] + self.impulses_mv[i-1][j]) / 2
                self.v_mv[j] = self.v_mv[j] / (sqrt(1 + temp))  # получили вектор скорости
            for j in range(3):
                self.force_mv[j] = self.currE_mv[j] + cross(self.v_mv, self.currH_mv, j)  # сила Лоренца
            K = (1 + temp) * 1.18 * 1e-8 * (dot(self.force_mv, self.force_mv) - (dot(self.v_mv, self.force_mv))**2)  #буква К
            if (self.radiation == 0):
                K = 0
            for j in range(3):
                self.impulses_mv[i][j] = self.impulses_mv[i][j] - dt * K * self.v_mv[j]
            for j in range(3):
                self.coord_mv[i][j] = self.coord_mv[i-1][j] + self.impulses_mv[i][j] * dt / (sqrt(1 + temp)) #координаты
        return self.impulse, self.coord
        
cdef class Particle:  # класс частицы
    cdef:
        double [:] x_mv, p_mv
        np.ndarray x, p
        Electromagnetic field

    def __init__(self, x, p, field):
        self.x = np.array(x).astype(np.double)  # иначе вектор [0, 0, 0] не работает
        self.p = np.array(p).astype(np.double)
        self.x_mv = self.x
        self.p_mv = self.p
        self.field = field
        
cdef class Intensity:
    cdef:
        double phi, theta
        double [:, :] trajec_mv, momen_mv
        double [:] n_mv, omega_mv, time_mv
        double [:] r_mv, r1_mv, v_mv, v1_mv, dv_mv, v_ave_mv
        double [:] result_mv, temp_mv, temp1_mv
        double ksi, ksi1, dksi, ksi_ave
        double [:] vect1_mv, vect2_mv, vect3_mv, vect4_mv, vect5_mv, vect6_mv 
        np.ndarray trajec, momen, n, omega, time
        np.ndarray r, r1, v, v1, dv, v_ave
        np.ndarray result, temp, temp1
        np.ndarray vect1, vect2, vect3, vect4, vect5, vect6
        int size, size_time
    
    def __init__(self, np.ndarray x, np.ndarray p):
        self.trajec = x  # координаты
        self.momen = p  # импульсы
        self.trajec_mv = self.trajec
        self.momen_mv = self.momen
        self.size = 1000  # столько интегралов
        self.size_time = 10000  # разбиение интеграла
        
        self.n = np.zeros(3)
        self.n_mv = self.n
        self.omega = np.linspace(0, 5, self.size)
        self.omega_mv = self.omega
        
        self.r = np.zeros(3)
        self.r1 = np.zeros(3)
        self.v = np.zeros(3)
        self.v1 = np.zeros(3)
        self.dv = np.zeros(3)
        self.v_ave = np.zeros(3)
        
        self.r_mv = self.r
        self.r1_mv = self.r1
        self.v_mv = self.v
        self.v1_mv = self.v1
        self.dv_mv = self.dv
        self.v_ave_mv = self.v_ave
        
        self.temp = np.zeros(3)
        self.temp1 = np.zeros(3)
        self.temp_mv = self.temp
        self.temp1_mv = self.temp1
        self.result = np.zeros(self.size)
        self.result_mv = self.result
        
        self.vect1 = np.zeros(3)
        self.vect2 = np.zeros(3)
        self.vect3 = np.zeros(3)
        self.vect4 = np.zeros(3)
        self.vect5 = np.zeros(3)
        self.vect6 = np.zeros(3)
        self.vect1_mv = self.vect1
        self.vect2_mv = self.vect2
        self.vect3_mv = self.vect3
        self.vect4_mv = self.vect4
        self.vect5_mv = self.vect5
        self.vect6_mv = self.vect6
        
        self.time = np.linspace(0, 600, self.size_time * 10)
        self.time_mv = self.time
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False) 
    cpdef np.ndarray intensity(self, phi, theta):
        cdef double dt
        cdef double gam, gam1
        cdef int i, j, k
        dt = self.time_mv[1] - self.time_mv[0]

        self.phi = phi
        self.theta = theta
        self.n = np.zeros(3)
        self.n_mv = self.n
        self.n_mv[0] = cos(self.phi) * sin(self.theta)
        self.n_mv[1] = sin(self.phi) * sin(self.theta)
        self.n_mv[2] = cos(self.theta)

        #self.n_mv[0] = 1
        with nogil:
            for i in prange(1, self.size, 1, num_threads=4):  # цикл по числу интегралов
                for k in range(3):
                    self.vect1_mv[k] = 0
                    self.vect2_mv[k] = 0
                for j in range(self.size_time - 1):  # цикл по итерациям одного интеграла
                    for k in range(3):
                        self.r_mv[k] = self.trajec_mv[j][k]
                        self.r1_mv[k] = self.trajec_mv[j+1][k]
                        self.v_mv[k] = self.momen_mv[j][k]
                        self.v1_mv[k] = self.momen_mv[j+1][k]
                    self.ksi = self.time_mv[j] - dot(self.n_mv, self.r_mv)
                    self.ksi1 = self.time_mv[j+1] - dot(self.n_mv, self.r1_mv)
                    self.dksi = self.ksi1 - self.ksi
                    self.ksi_ave = (self.ksi1 + self.ksi) / 2
                    
                    gam = 1 + dot(self.v_mv, self.v_mv)
                    gam1 = 1 + dot(self.v1_mv, self.v1_mv)
                    for k in range(3):
                        self.v_mv[k] = self.v_mv[k] / sqrt(gam)
                        self.v1_mv[k] = self.v1_mv[k] / sqrt(gam1) 
                    for k in range(3):
                        self.dv_mv[k] = self.v1_mv[k] - self.v_mv[k]
                        self.v_ave_mv[k] = (self.v1_mv[k] + self.v_mv[k]) / 2
                    for k in range(3):
                        self.temp_mv[k] = cos(self.omega_mv[i] * self.ksi_ave) * dt / self.dksi * 2 * self.v_ave_mv[k] * sin(self.omega_mv[i] * self.dksi / 2)  # 1 часть
                        self.temp_mv[k] = self.temp_mv[k] - sin(self.omega_mv[i] * self.ksi_ave) * dt / self.dksi * self.dv_mv[k] * ((sin(self.omega_mv[i] * self.dksi / 2) / (self.omega_mv[i] * self.dksi / 2))  - cos(self.omega_mv[i] * self.dksi / 2))
                        #действительная часть вектора. Теперь надо найти мнимую часть этого вектора
                        self.temp1_mv[k] = cos(self.omega_mv[i] * self.ksi_ave) * dt / self.dksi * self.dv_mv[k] * ((sin(self.omega_mv[i] * self.dksi / 2) / (self.omega_mv[i] * self.dksi / 2))  - cos(self.omega_mv[i] * self.dksi / 2))
                        self.temp1_mv[k] = self.temp1_mv[k] + sin(self.omega_mv[i] * self.ksi_ave) * dt / self.dksi * 2 * self.v_ave_mv[k] * sin(self.omega_mv[i] * self.dksi / 2)
                        #получили мнимую часть вектора.
                        #результат - сумма квадратов
                    for k in range(3):
                        self.vect1_mv[k] = self.vect1_mv[k] + self.temp_mv[k]
                        self.vect2_mv[k] = self.vect2_mv[k] + self.temp1_mv[k]
                    #просуммировали дей. и мнимую части
                #теперь у нас есть 2 вектора. Дей. и мнимая части.
                for k in range(3):
                    self.vect3_mv[k] = cross(self.n_mv, self.vect1_mv, k)
                    self.vect4_mv[k] = cross(self.n_mv, self.vect2_mv, k)
                for k in range(3):
                    self.vect5_mv[k] = cross(self.n_mv, self.vect3_mv, k)
                    self.vect6_mv[k] = cross(self.n_mv, self.vect4_mv, k)
                #финальный вектор из 2 частей. Ответ - сумма квадратов
                self.result_mv[i] = (1.0 / (4.0 * pi * pi)) *(dot(self.vect5_mv, self.vect5_mv) + dot(self.vect6_mv, self.vect6_mv)) 
        return self.result
        
# Инструкция по работе с данными классами и методами.
# cdef dot(a, b) - на вход подается 2 вектора memoryview, на выходе получаем 
# скалярное произведение.
# cdef cross(a, b, j) - на вход 2 вектора memoryview, а также координата нужной
# нам компоненты векторного произведения. Для возврата всего вектора использовать
# цикл.
# cdef norm(a) - возвращает норму вектора

# Как работать с классом полей Electromagnetic. При инициализации объекта можно
# указать амплитуду поля, волновой вектор k и длительность лазерного импульса.

# Как получить векторы полей. Для этого предназначены функции e_field и h_field.
# На вход подавать:
#     1. тип поля ('crossfield' - скрещенные поля, 'wavefield' - поле
#                             электромагнитной волны, 'laserpulse' - лазерный)
#     2. вектор координат r -  та точка, где будет рассчитано поле
#     3.  t -  момент времени, в который будет рассчитано поле
# Для скрещенных полей по умолчанию Е направлено по оси х, а Н по оси у

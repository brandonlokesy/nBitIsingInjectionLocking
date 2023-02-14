import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import datetime
import pandas as pd
from natsort import natsorted
import os
from os import listdir
from os.path import isfile, join

def format_scientific(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

def dwavelength(df, wavelength0):
    c=3e8
    f0= c/wavelength0
    return (c/f0) - (c/(f0-df))

def dfrequency(dwavelength, wavelength0):
    c=3e8
    f0=c/wavelength0
    return f0 - ((1/f0) - (dwavelength/c))**(-1)

def basic_plot(filename, norm=False):
    t, Ex, Ey, phix, phiy, N, m = np.loadtxt(f'./Simulation Data/1550nm/{filename}.dat', unpack=True)
    if norm:
        Ex = Ex/np.max(Ex)
        Ey = Ey/np.max(Ey)
    lw=1
    fig, ax = plt.subplots()
    ax.plot(t/1e-9, Ex, 'b', linewidth=lw, label = '$E_x$')
    ax.plot(t/1e-9, Ey, 'r', linewidth=lw, label = '$E_y$')
    ax.set_ylabel('Amplitude (a.u)')
    ax.set_xlabel('Time (ns)')
    ax.legend()
    plt.show()

def dSdt(var, t, params, bits):
    Ex = var[:bits]
    Ey = var[bits:2*bits]
    phix = var[2*bits:3*bits]
    phiy = var[3*bits:4*bits]
    N = var[4*bits:5*bits]
    m = var[5*bits:6*bits]

    ExNew = np.zeros(bits)
    EyNew = np.zeros(bits)
    phixNew = np.zeros(bits)
    phiyNew = np.zeros(bits)
    NNew = np.zeros(bits)
    mNew = np.zeros(bits)

    kappa, yp, ya, ys, y, alpha, Kinj, bsp, eta, dw, Einjx, Einjy, delta, xi, AMatrix, deltaCouplingMatrix = params

    wx = alpha*ya - yp
    wy = yp- alpha*ya

    sigmajix_matrix = np.zeros(shape = (bits,bits))
    sigmajiy_matrix = np.zeros(shape = (bits,bits))
    for j in range(bits):
        for i in range(bits):
            sigmajix_matrix[j][i] = wx + wy + phix[j] - phix[i] + deltaCouplingMatrix[j][i]
            sigmajiy_matrix[j][i] = wx + wy + phiy[j] - phiy[i]
        sigmajix_matrix[j][j] = 0
        sigmajiy_matrix[j][j] = 0

    Ex_mutual_injection_lock = Kinj * np.matmul(np.multiply(AMatrix(t), np.cos(sigmajix_matrix)), Ex)
    Ey_mutual_injection_lock = Kinj * np.matmul(np.multiply(AMatrix(t), np.cos(sigmajiy_matrix)), Ey)
    phix_mutual_injection_lock = Kinj * np.matmul(np.multiply(AMatrix(t), np.sin(sigmajix_matrix)), Ex)
    phiy_mutual_injection_lock = Kinj * np.matmul(np.multiply(AMatrix(t), np.sin(sigmajiy_matrix)), Ey)
    # print(sigmajix_matrix)

    # sigmax21 = wx + wy + phix[1] - phiy[0]
    # print(f"Other sigma is {sigmax21}")

    for n in range(bits):
        deltax_n = wy*t - phix[n] + delta
        deltay_n = wx*t - phiy[n]
        deltaphi_n = 2*wy*t + phiy[n] - phix[n]

        ExNew[n] = kappa * ( (N[n]-1)*Ex[n] - m[n]*Ey[n] * (np.sin(deltaphi_n) + alpha*np.cos(deltaphi_n))) - ya*Ex[n] + Kinj*Einjx(t)*np.cos(deltax_n) #+ np.sqrt(bsp*(N[n]+m[n]))*xi(t)

        EyNew[n] = kappa * ( (N[n]-1)*Ey[n] + m[n]*Ex[n] * (alpha * np.cos(deltaphi_n) - np.sin(deltaphi_n))) + ya*Ey[n] + Kinj*Einjy(t)*np.cos(deltay_n) #+ np.sqrt(bsp*(N[n]-m[n]))*xi(t)

        phixNew[n] = kappa * ( alpha*(N[n]-1) + m[n] * (Ey[n]/Ex[n]) * (np.cos(deltaphi_n) - alpha*np.sin(deltaphi_n))) - dw(t) - alpha*ya + Kinj * (Einjx(t)/Ex[n]) * np.sin(deltax_n)

        phiyNew[n] = kappa * ( alpha*(N[n]-1) - m[n] * (Ex[n]/Ey[n]) * (alpha* np.sin(deltaphi_n) + np.cos(deltaphi_n))) - dw(t) + alpha*ya + Kinj * (Einjy(t)/Ey[n]) * np.sin(deltay_n)

        NNew[n] = -y*( N[n]*(1+ np.power(Ex[n],2) + np.power(Ey[n],2)) - eta - 2*m[n]* Ey[n]* Ex[n] * np.sin(deltaphi_n))

        mNew[n] = -ys*m[n] - y*(m[n]*(np.power(Ex[n],2) + np.power(Ey[n],2))) + 2*y*N[n]*Ey[n]*Ex[n]*np.sin(deltaphi_n)

    ExNew = ExNew + Ex_mutual_injection_lock
    EyNew = EyNew + Ey_mutual_injection_lock
    phixNew = phixNew + phix_mutual_injection_lock * (1/Ex)
    phiyNew = phiyNew + phiy_mutual_injection_lock * (1/Ey)
    f = np.concatenate((ExNew, EyNew, phixNew, phiyNew, NNew, mNew))
    return f

def solve(filename, master = False, mutual = False, noise = False):
    # ! General Parameters
    kappa = 125e9
    yp = 192e9
    ya = 0.02e9
    ys = 1000e9
    y = 0.67e9
    alpha = 3
    Kinj = 35.5e9
    bsp = 10e-5
    eta = 3.4 #From Al-Seyab, R., K. Schires, A. Hurtado, I. D. Henning, and M. J. Adams. ‘Dynamics of VCSELs Subject to Optical Injection of Arbitrary Polarization’. IEEE Journal of Selected Topics in Quantum Electronics 19, no. 4 (July 2013): 1700512–1700512. https://doi.org/10.1109/JSTQE.2013.2239614.

    # ! Solving Parameters
    abserr = 1.0e-9
    relerr = 1.0e-7
    stoptime = 10.0
    numpoints = 250
    del_t = 1e-12
    t = np.arange(0,40e-9,del_t)

    # ! Bits and Initial Values
    bits = 2
    EConstant = 0.001
    Ex = EConstant * np.ones(bits)
    Ey = EConstant * np.ones(bits)
    phix = np.zeros(bits)
    phiy = np.zeros(bits)
    N = np.zeros(bits)
    m = np.zeros(bits)

    # ! Injection Parameters
    def inj_w(t):
        if t>20e-9:
            return 1.956450e+11
        return 0

    dw = inj_w
    delta = 0

    def Einjx_func(t):
        if t>20e-9:
            return 0.2
        return 0

    def Einjy_func(t):
        if t>20e-9:
            return 0.2
        return 0

    def gaussian_noise(t):
        return np.random.normal(0,1)
    
    Einjx = Einjx_func
    Einjy = Einjy_func

    if not master:
        Einjx = lambda x: 0
        Einjy = lambda x: 0
    xi = gaussian_noise
    if not noise:
        xi = lambda x:0

    # ! Coupling Parameters
    deltaCouplingMatrix = np.zeros(shape=(bits, bits))

    def coupling(t):
        if t < 30e-9:
            return np.zeros(bits)
        A = np.ones(shape = (bits, bits))
        np.fill_diagonal(A, 0)
        return A

    AMatrix = coupling
    if not mutual:
        AMatrix =lambda x: 0
    
    # ! Initialise arguments
    params = kappa, yp, ya, ys, y, alpha, Kinj, bsp, eta, dw, Einjx, Einjy, delta, xi, AMatrix, deltaCouplingMatrix
    var = np.array([Ex, Ey, phix, phiy, N, m], dtype = np.longdouble)
    solve_params = abserr, relerr, stoptime, numpoints, del_t, t


    # print(var.flatten())
    # ! Solve
    print("### Start ###")
    wsol, output = odeint(dSdt, var.flatten(), t, args = (params, bits,),
        atol = abserr, rtol = relerr, full_output=1)
    data = np.hstack((np.reshape(t, (np.shape(wsol)[0],1)), wsol))
    np.savetxt(f'./Simulation Data/nbits/{filename}.dat', data)
    print("### END ###")

if __name__ == '__main__':
    master = True
    mutual = True
    noise = False
    filename = 'test_2bits_master_coupled'
    solve(filename = filename, master = master, mutual = mutual, noise = noise)
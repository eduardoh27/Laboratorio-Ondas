import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def main():
    plt.style.use(['science','no-latex'])
    
    #np.seterr(divide='ignore', invalid='ignore')
    #test()
    #una()
    doble()
    None
#Actividad 2

def doble():
    
    #doble = pd.read_csv("doble rendija.csv")
    #angulo = np.around(doble["theta (rad)"].to_numpy(), 5)
    #voltaje = doble["Voltaje (V)"].to_numpy()
    
    x_values = [0,0.008761851,0.017457019,0.026019328,0.034383615,0.042486222,
            0.050265482,0.057662193,0.064620059,0.071086127,0.077011187,
            0.082350145,0.087062369,0.091111996,0.094468206,0.097105455,
            0.099003674,0.100148414,0.100530965]
    corriente_values = np.array([40.,30.,15.,5.,9.,12.,9.,8.,4.,3.,
                                 3.,3.,3.,3.,2.,2.,1.,0.,0.])
    corriente_values*=1e-6
        
    d = 0.055
    a = 0.032
    c = d/a
    
    
    plt.figure()
    #plt.scatter(angulo,voltaje, color='k') 
    #plt.errorbar(angulo, voltaje, xerr=0.00001, yerr = 0.005, fmt=" ")
    plt.scatter(x_values,corriente_values, color='k') 
    
    #plt.show()
    #thetax = np.linspace(-0.005,0.005,100)
    x_valuesx = np.linspace(0,0.1,100)

    def f1_ (theta,A,B,C):
        return A*(np.cos(B*np.sin(theta)))**2 * ((np.sin(C*np.sin(theta)))/(C*np.sin(theta)))**2

    def f1 (x,A,B):
        return A * (np.cos(c*x/B))**2 * (np.sin(x/B))**2 / (x/B)**2
    #A * (np.sin(x/B))**2 / (x/B)**2

    #popt,pcov = curve_fit(f1,angulo, voltaje, p0=[2.5,2300,500])
    popt,pcov = curve_fit(f1,x_values, corriente_values, p0=[4e-5,2.71438030e-02])
    
    #bounds=([1,2000,500],[3,3000,600])
    
    #yajuste = f1(thetax, popt[0], popt[1], popt[2])   
    #plt.plot(thetax,yajuste)
    
    yajuste = f1(x_valuesx, popt[0], popt[1])
    plt.plot(x_valuesx, yajuste, label = r'$\lambda = 2.71438030e-02 \pm 3.41296078e-03$')
    
    print(popt)
    print(np.sqrt(np.diag(pcov)))
    
    
    
    plt.title("Doble rendija")
    plt.legend()
    plt.xlabel("x (m)")
    plt.ylabel("Corriente (A)")
    plt.show()
    
    """
    print(popt) #A, B y C (importan B y C)
    print("")
    print(np.sqrt(np.diag(pcov))) #Incertidumbres A, B y C
    print("")
    print("")
    """


def una():

    #una rendija

    plt.figure()
    
 
    x_values = [0.002741,    0.005478661,    0.008209646,    0.01093063,
    0.013638296,    0.016329346,    0.019000501,    0.021648507,
    0.024270138,    0.026862199,    0.029421533,    -0.002741,   -0.005478661,
    -0.008209646,    -0.01093063,    -0.013638296,    -0.016329346,
    -0.019000501,  -0.021648507,    -0.024270138]
    
    corriente_values = np.array([36.,35.,34.,32.,30.,30.,28.,26.,20.,15.,7.,36.,35.,
    32.,28.,27.,25.,20.,12.,10.])
    corriente_values*=1e-6


    #ang = np.around(simple["theta"].to_numpy(),5) # ang (xdata)
    #volt = simple["voltaje"].to_numpy() # volt (ydata)
    
    plt.scatter(x_values, corriente_values, color="k")

    x_valuesx = np.linspace(-0.025, 0.0295, 100)
    #angx = np.linspace(-0.008, 0.008, 100)
    
    def f2_ (theta, A, B):
        return A*(np.sin(B*np.sin(theta)))**2 / (B*np.sin(theta))**2

    def f2(x, A, B):
        return A * (np.sin(x/B))**2 / (x/B)**2
    
    def f21(x, B):
        return (np.sin(x/B))**2 / (x/B)**2

    
    #ajuste, cov = curve_fit(f2, ang, volt, p0=[0.25, 300])
    ajuste, cov = curve_fit(f2, x_values, corriente_values)
    
    print(ajuste) #A, B (importa B)
    print("")
    print(np.sqrt(np.diag(cov))) #Incertidumbres A y B

    #plt.plot(angx, f2(angx, ajuste[0], ajuste[1]))
    
    #plt.plot(x_valuesx, f2(x_valuesx, ajuste[0], ajuste[1]))
    plt.plot(x_valuesx, f2(x_valuesx, ajuste[0], ajuste[1]), label = r"$\lambda = 1.61438030e-02 \pm 9.00397757e-04$")
    
    plt.title("Una rendija")
    plt.legend()
    plt.xlabel("x (m)")
    plt.ylabel("Corriente (A)")
    plt.show()

def test():
    
    d = 0.055
    a = 0.032
    c = d/a
    
    plt.figure()
    x_valuesx = np.linspace(0, 0.1, 100)
    
    def f1 (x,A,B):
        return A * (np.cos(c*x/B))**2 * ((np.sin(x/B))/(x/B))**2
    
    print(f1(x_valuesx, 1, 1))   
    plt.plot(x_valuesx, f1(x_valuesx, 0.0001, 0.01))
    plt.show()

if __name__ == '__main__':
    main()
    



# 0.0435

# 
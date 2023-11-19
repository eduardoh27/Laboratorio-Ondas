import numpy as np
import matplotlib.pyplot as plt

def main():
    frecs_list = np.linspace(5500,9500,40+1)
    
    fase_list = np.array([38.7,36.7,36.3,34,31.2,30.3,26.1,25,23.2,20.4,
                18.8,16.6,14.9,12.6,10.8,8.09,7.68,6.24,4.74,
                3.21,1.08,-1.09,-3.32,-4.49,-6.35,-8.5,-10.4,
                -13,-16,-17.9,-19.5,-21,-22,-22.9,-24.2,-25.8,
                -26.6,-27,-28.8,-29.8,-31.4])
    
    vpp1_list = np.array([3.64,3.72,3.76,3.84,3.92,3.96,4,4.04,4.08,4.12,
                4.16,4.2,4.24,4.28,4.28,4.32,4.32,4.32,4.32,4.36,
                4.36,4.32,4.32,4.28,4.28,4.28,4.28,4.24,4.2,4.16,
                4.12,4.12,4.08,4.08,4.04,4,3.96,3.92,3.88,3.84,3.8])
    
    vpp2_list = np.array([4.88,4.88,4.88,4.88,4.88,4.88,4.88,4.8,4.8,4.72,4.72,
                4.72,4.72,4.72,4.72,4.72,4.72,4.72,4.72,4.72,4.72,4.72,
                4.72,4.72,4.72,4.72,4.72,4.72,4.72,4.72,4.72,4.72,4.72,
                4.72,4.72,4.8,4.8,4.8,4.88,4.88,4.88])

    """
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(fase, frecs)
    ax.plot(frecs, fase)
    plt.show()
    """
    
    omegas_list = frecs_list*(2*np.pi)
    print(omegas_list)
    
if __name__ == "__main__":
    main()
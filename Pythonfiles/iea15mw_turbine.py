from py_wake import np
import pandas as pd
import numpy as np
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines._wind_turbines import WindTurbine

# ---------------------------- INFO -----------------------------------#
# This file defines the IEA 15 MW turbine class
# ---------------------------------------------------------------------#



# Read your CSV data (make sure to adjust the path)
data = pd.read_csv("../Data/IEA_Reference_15MW_240.csv")

# Extract relevant columns
u = data['Wind Speed [m/s]'].values
p = data['Power [kW]'].values
ct = data['Ct [-]'].values

# Create power curve with loss
power_curve = np.column_stack((u, p))  # Combine wind speed and power


# Create Ct curve
ct_curve = np.column_stack((u, ct))  # Combine wind speed and Ct


class iea15mw(WindTurbine):
    '''
    Data from: IEA 15 MW reference Turbine
    
    '''

    def __init__(self, method='linear'):
        u, p = power_curve.T
        WindTurbine.__init__(
            self,
            'iea15mw',
            diameter=240, # m
            hub_height=150, # m
            powerCtFunction=PowerCtTabular(u, p * 1000, 'w', ct_curve[:, 1], ws_cutin=3, ws_cutout=25,
            
                                           ct_idle=0.059, method=method))


# ct_idle is represents the thrust coefficient of a wind turbine in idle mode, meaning it is not producing power but still affecting the wind flow.

def main():
    wt = iea15mw()

    print('Diameter', wt.diameter())
    print('Hub height', wt.hub_height())
    ws = np.arange(3, 25)
    import matplotlib.pyplot as plt
    plt.plot(ws, wt.power(ws), '.-', label='power [W]')
    c = plt.plot([], label='ct')[0].get_color()
    plt.legend()
    ax = plt.twinx()
    ax.plot(ws, wt.ct(ws), '.-', color=c)

    plt.show()


if __name__ == '__main__':
    main()


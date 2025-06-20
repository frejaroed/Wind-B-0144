# ----------------------------------------
# Title: Smart Start AEP
# Author: Freja Roed Søndergaard & Christian Bonde
# Student ID: s223847 & s223865
# Date: June 2025
# Description: This script performs a smart start optimization maximizing AEP. 
# ----------------------------------------
#%%---------------------------------------------------------------------------------#
#                       IMPORT ALL REQUIRED PACKAGES & FUNCTIONS                    #
#-----------------------------------------------------------------------------------#
#%% IMPORT ALL REQUIED PACKAGES
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import os
import py_wake
import seaborn as sns

# IMPORT ALL REQUIRED FUNCTIONS
from matplotlib.patches import Circle
from scipy.interpolate import Rbf
from openmdao.api import n2
from topfarm._topfarm import TopFarmProblem, TopFarmGroup
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.plotting import XYPlotComp
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from topfarm.constraint_components.boundary import XYBoundaryConstraint, CircleBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from py_wake import NOJ, BastankhahGaussian
from py_wake.utils.gradients import autograd
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake.site.xrsite import GlobalWindAtlasSite
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.drivers.random_search_driver import RandomizeTurbinePosition_Circle
from topfarm.easy_drivers import EasyRandomSearchDriver
from scipy.spatial import distance
from shapely.geometry import Point, Polygon
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.site.xrsite import GlobalWindAtlasSite
from py_wake.examples.data.iea22mw.iea_22_rwt import IEA_22MW_280_RWT, IEA_22MW_280_RWT_ct_curve, IEA_22MW_280_RWT_power_curve
from iea15mw_turbine import iea15mw

#%% ---------------------------------------------------------------------------------#
#                                DEFINE GENERIC WIND TURBINES                        #
#------------------------------------------------------------------------------------#
# Define wind turbine and rated power
def set_wt(MW, use_ref=False):
    Ti = 0.1

    # Generic turbines
    gen_wt8 = GenericWindTurbine('Gen8MW', diameter=173.1, hub_height=117, power_norm=8000, turbulence_intensity=Ti, ws_cutin=3)
    gen_wt10 = GenericWindTurbine('Gen10MW', diameter=193.5, hub_height=127, power_norm=10000, turbulence_intensity=Ti, ws_cutin=3)
    gen_wt12 = GenericWindTurbine('Gen12MW', diameter=212, hub_height=136, power_norm=12000, turbulence_intensity=Ti, ws_cutin=3)
    gen_wt15 = GenericWindTurbine('Gen15MW', diameter=237, hub_height=149, power_norm=15000, turbulence_intensity=Ti, ws_cutin=3)
    gen_wt17 = GenericWindTurbine('Gen17MW', diameter=252.3, hub_height=156, power_norm=17000, turbulence_intensity=Ti, ws_cutin=3)
    gen_wt20 = GenericWindTurbine('Gen20MW', diameter=273.7, hub_height=167, power_norm=20000, turbulence_intensity=Ti, ws_cutin=3)
    gen_wt22 = GenericWindTurbine('Gen22MW', diameter=287, hub_height=174, power_norm=22000, turbulence_intensity=Ti, ws_cutin=3)
    # Reference turbines
    ref_wts = {
        10: DTU10MW(),
        15: iea15mw(),
        22: IEA_22MW_280_RWT()
    }

    if use_ref:
        if MW in ref_wts:
            return ref_wts[MW], MW
        else:
            print(f"No reference turbine available for {MW} MW.")
            return None, None
    else:
        if MW == 8:
            return gen_wt8, 8
        elif MW == 10:
            return gen_wt10, 10
        elif MW == 12:
            return gen_wt12, 12
        elif MW == 15:
            return gen_wt15, 15
        elif MW == 17:
            return gen_wt17, 17
        elif MW == 20:
            return gen_wt20, 20
        elif MW == 22:
            return gen_wt22, 22
        else:
            print(f"No generic wind turbine defined for {MW} MW.")
            return None, None


# Example usage:
wt, rated_power = set_wt(15)              # generic 15 MW
wt_ref, rated_power = set_wt(15, True)    # reference 15 MW
wt_none, rated_power = set_wt(12, True)   # None, no reference for 12 MW
#%% ---------------------------------------------------------------------------------#
#                                DEFINE FUNCTIONS                                    #
#------------------------------------------------------------------------------------#

# -------------------------- Define function to get site data------------------------#
def get_site(site_name, wt, rated_power):
    # Construct dynamic file path for the selected site
    base_dir = os.path.dirname(f'{site_name}.ipynb')  # Adjust this path if necessary
    data_file = f'boundary_{site_name}.csv'
    data_path = os.path.join(base_dir, '..', 'Data', data_file)

    # Import site boundaries
    EIB_data = pd.read_csv(data_path)
    EIB_boundary = EIB_data.to_numpy()

    # Define site boundary coordinates from unique points
    x_site = EIB_boundary[:, 0]
    y_site = EIB_boundary[:, 1]

    if site_name == 'EIB1N':
        site_lat, site_lon = 54.9998268, 14.3235229
        wf_capacity = 750  # Wind farm capacity for EIB1N
        x_points = 20
        y_points = 30
    elif site_name == 'EIB1S':
        site_lat, site_lon = 54.8810215, 14.1693805
        wf_capacity = 750  # Wind farm capacity for EIB1S
        x_points = 25
        y_points = 20
    elif site_name == 'EIB2':
        site_lat, site_lon = 54.7079033, 14.707903
        wf_capacity = 1500  # Wind farm capacity for EIB2
        x_points = 30
        y_points = 30
    else:
        raise ValueError(f"Unknown site name: {site_name}")

    site = GlobalWindAtlasSite(
        lat=site_lat,
        long=site_lon,
        height=wt.hub_height(),
        ti=0.1,
        roughness=0.0002
    )

    # Set number of turbines
    n_wt = int(wf_capacity / rated_power)  # Number of wind turbines

    x = np.linspace(np.min(x_site), np.max(x_site), x_points)
    y = np.linspace(np.min(y_site), np.max(y_site), y_points)
    XX, YY = np.meshgrid(x, y)

    return EIB_boundary, x_site, y_site, site, n_wt, x_points, y_points, XX, YY


# ------------------------------- Define wake model ------------------------------#
def set_wake_model(wake_model_name, site, wt):
    if wake_model_name == 'NOJ':
        return NOJ(site, wt)
    elif wake_model_name == 'BastankhahGaussian':
        return BastankhahGaussian(site, wt)
    else:
        raise ValueError(f"Unknown wake model name: {wake_model_name}")

#%% ---------------------------------------------------------------------------------#
#                            DEFINE OPTIMIZATION PROBLEM                             #
#------------------------------------------------------------------------------------#
def get_problem(wt, EIB_boundary, wf_model, n_wt):
    
    spacing_constraint = 4 * wt.diameter()  # Minimum distance between wind turbines

    x_init = np.random.uniform(EIB_boundary[:, 0].min(), EIB_boundary[:, 0].max(), size=n_wt)
    y_init = np.random.uniform(EIB_boundary[:, 1].min(), EIB_boundary[:, 1].max(), size=n_wt)

    problem = TopFarmProblem(
        design_vars={'x': x_init, 'y': y_init},
        driver=EasyScipyOptimizeDriver(optimizer='SLSQP', disp=False),
        cost_comp=PyWakeAEPCostModelComponent(wf_model, n_wt, grad_method=autograd, objective=True), 
        constraints=[XYBoundaryConstraint(np.array([EIB_boundary[:, 0], EIB_boundary[:, 1]]).T, boundary_type='polygon'),
                    SpacingConstraint(spacing_constraint)],
        plot_comp=XYPlotComp())

    return problem

#%% ----------------------- Evaluate problem (no optimization yet) ---------------------%%#
#cost_init, state_init = problem.evaluate(dict(zip('xy', [x_init, y_init])))
#print(f"initial AEP: ", abs(cost_init))
#%% ---------------------------------------------------------------------------------#
#                               APPLY SMART START-START                              #
#------------------------------------------------------------------------------------#
# Initialize an empty DataFrame to store results
results_df = pd.DataFrame(columns=['turbine [MW]', 'site', 'seed', 'AEP [GWh]', 'AEP without wake loss [GWh]', 'Wake loss [%]'])

# Define turbines and sites to iterate over
turbines = [8,10,12,15,17,20,22]  # Turbine capacities in MW
sites = ['EIB1N','EIB1S','EIB2']      # Site names
seeds = range(1, 11)    # Seeds from 1 to 10

# Nested loops for turbines, sites, and seeds
for turbine in turbines:
    wt, rated_power = set_wt(turbine, use_ref=True)  # Set turbine
    if wt is None:  # Skip if turbine is not defined
        continue

    for site_name in sites:
        # Get site-specific data
        EIB_boundary, x_site, y_site, site, n_wt, x_points, y_points, XX, YY = get_site(site_name, wt, rated_power)
        # Set wake model
        wf_model = set_wake_model('NOJ', site, wt)  # Using NOJ as the wake model
        # Define the optimization problem
        problem = get_problem(wt, EIB_boundary, wf_model, n_wt)
        
        for seed in seeds:
            # Perform smart start optimization
            problem.smart_start(
            XX,
            YY,
            ZZ=problem.cost_comp.get_aep4smart_start(),
            min_space=4 * wt.diameter(),  # Spacing constraint
            random_pct=0.1,
            seed=seed, 
            plot=False
            )
            
            # Evaluate the problem
            cost_smart, state_smart = problem.evaluate()
            sim_res = wf_model(state_smart['x'], state_smart['y'])
            aep_without_wake_loss = sim_res.aep(with_wake_loss=False).sum().data
            # Append the results to the DataFrame
            results_df = pd.concat([results_df, pd.DataFrame({
                'turbine [MW]': [turbine],
                'site': [site_name],
                'seed': [seed],
                'AEP [GWh]': [abs(cost_smart)],
                'AEP without wake loss [GWh]': [aep_without_wake_loss],
                'Wake loss [%]': [((aep_without_wake_loss - abs(cost_smart))/aep_without_wake_loss) * 100]
            })], ignore_index=True)
    print()
    print("----------------------------------------------")
    print(f"Completed turbine {turbine}MW")
    print("----------------------------------------------")
    print()

# Print the resulting DataFrame
print(results_df)

#%% ---------------------------------------------------------------------------------#
#                               PLOT THE RESULTS                                     #
# -----------------------------------------------------------------------------------#

# Define spacing constant (4*Diameter)
n_spacing=4

def plot_state_with_seabed(state_smart, site_name, turbine, seabed_data_path=None, boundary_data_path=None, output_unit='M€', cost=None, cost_key='Foundation Cost', use_ref=False):
    # Load seabed data
    plt.rcParams.update({'font.size': 17})
    if seabed_data_path is None:
        seabed_data_path = os.path.join('..', 'Data', f'seabed_downsampled_{site_name}.csv')
    df_seabed = pd.read_csv(seabed_data_path)

    if boundary_data_path is None:
        boundary_data_path = os.path.join('..', 'Data', f'boundary_{site_name}.csv')
    df_boundary = pd.read_csv(boundary_data_path)

    # Determine figure size based on site_name
    if site_name == 'EIB1N':
        fig_size = (8, 10)
        marker_size = 50
    elif site_name == 'EIB1S':
        fig_size = (10, 6)
        marker_size = 50
    elif site_name == 'EIB2':
        fig_size = (10, 7)
        marker_size = 50
    else:
        fig_size = (10, 8)  # Default size

    # Initialize the plot
    fig, ax = plt.subplots(figsize=fig_size)  # Use ax for consistent plotting
    # Plot seabed data
    scatter = ax.scatter(df_seabed['utm_x'], df_seabed['utm_y'], c=df_seabed['elevation'], cmap='viridis', s=marker_size, marker='s')
    cbar = fig.colorbar(scatter, ax=ax)  # Add colorbar
    cbar.set_label('Seabed Elevation [m]')

    # Plot boundary as a polygon
    ax.plot(df_boundary['X'], df_boundary['Y'], color='black', linestyle='-', linewidth=1.5, label='Boundary')
    # Close the polygon by connecting the last point to the first
    ax.plot([df_boundary['X'].iloc[-1], df_boundary['X'].iloc[0]], 
            [df_boundary['Y'].iloc[-1], df_boundary['Y'].iloc[0]], 
            color='black', linestyle='-', linewidth=1.5)

    # Plot turbine positions
    ax.scatter(state_smart['x'], state_smart['y'], c='red', label=f'Turbine Positions', marker='x')

    # Set legend in the lower left corner with font size
    ax.legend(loc='upper left', prop={'size': 14})  # Position legend in the lower left corner

    # Get turbine object (to plot the diameter)
    wt = set_wt(turbine, use_ref)[0]

    # Get turbine positions
    x_turbines = state_smart['x']
    y_turbines = state_smart['y']

    # Add turbine constraint circles
    for x, y in zip(x_turbines, y_turbines):
        constraint_circle = Circle((x, y), (n_spacing/2) * wt.diameter(), edgecolor='black', facecolor='none', linestyle='--')
        ax.add_patch(constraint_circle)
    
    # Set plot labels and title
    ax.set_xlabel('UTM Easting [m]',fontsize=18)
    ax.set_ylabel('UTM Northing [m]')
    ax.locator_params(axis='x', nbins=7)
    # Set the number of ticks on x and y axes
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))  # 7 ticks on x-axis
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))  # 7 ticks on y-axis
    ax.tick_params(axis='both', which='major', labelsize=16)
    cost_text = f"{cost:.1f}" if cost is not None else "N/A"
    ax.set_title(f'{turbine}MW Turbines, {site_name} \n {cost_key}: {cost_text} {output_unit}') # Turbine Positions and Seabed Data for {site_name} \n
    ax.axis('equal')  # Ensure equal axis scaling
    plt.show()

plot_state_with_seabed(state_smart, site_name=site_name, turbine=turbine, seabed_data_path=None, output_unit='GWh', cost=abs(cost_smart), cost_key='AEP', use_ref=False)

#%% ----------------------------------------------------------------------------%%#
# Save the results to a CSV file
#---------------------------------------------------------------------------------#

# results_df.to_csv('smart_start_results_ref_01.csv', index=False)


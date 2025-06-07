
# ----------------------------------------
# Title: Smart Start LCOE and CO2 tax Analysis
# Author: Freja Roed Søndergaard & Christian Bonde
# Student ID: s223847 & s223865
# Date: June 2025
# Description: This script performs a smart start optimization for wind farm layout, focusing on minimizing LCOE and CO2 emissions. 
# ----------------------------------------

#%%----------------------------------------------------------------------------%%#
#                     IMPORT ALL REQUIED PACKAGES AND FUNCTIONS                  #
#---- ---------------------------------------------------------------------------#
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
from ed_win.wind_farm_network import WindFarmNetwork

#%%-----------------------------------------------------------------------------%%#
#                                    Define constants                             #
#---------------------------------------------------------------------------------#
r = 0.05 # interest rate 
LT = 30 # Life time of WF, years
cost_WT = (1.35+0.2+0.05+1.2)*10**6 # €/MW for the WT (old value was 1.1 M€) 0.2=project development cost, 0.05=array cables, 1.2=DC substation 
CRF = r / (1-(1+r)**(-LT)) # recovery factor
cost_fixed = 50*1000 # €/MW/year (old value was 40 k€)
cost_var = 5.0 # €/MWh (old value was 3 €/MWh)
CO2_tax = 125/7.45 # €/tCO2e (current value in Denmark, 2025)

EF_steel = 3.62 # tCO2e/t steel
EF_resin = 5.9 # tCO2e/t resin
EF_glass = 3.0 # tCO2e/t glass
EF_foam = 7.83 # tCO2e/t foam
EF_Cu = 4.1 # tCO2e/t Cu
EF_magnets = 27.6 # tCO2e/t magnets
EF_cable= 42.6941 # tCO2e/km cables


#%%-----------------------------------------------------------------------------%%#
#                       Define wind turbine and rated power                       #
#---------------------------------------------------------------------------------#
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


#%% -----------------------------------------------------------------------------%%#
#                       Define site and wind farm capacity                        #
#---------------------------------------------------------------------------------#
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

#%%-----------------------------------------------------------------------------%%#
#                               Define wake model                                #
#---------------------------------------------------------------------------------#
def set_wake_model(wake_model_name, site, wt):
    if wake_model_name == 'NOJ':
        return NOJ(site, wt)
    elif wake_model_name == 'BastankhahGaussian':
        return BastankhahGaussian(site, wt)
    else:
        raise ValueError(f"Unknown wake model name: {wake_model_name}")
#%%-----------------------------------------------------------------------------%%#
#                                Define relevant functions                         #
#---------------------------------------------------------------------------------#
# Calculate water depth
def water_depth_func(x, y, site_name, seabed_data_path=None):
    # Interpolate water depth from seabed elevation data
    if seabed_data_path is None:
        seabed_data_path = os.path.join('..', 'Data', f'seabed_downsampled_{site_name}.csv')
    df_seabed = pd.read_csv(seabed_data_path)
    points = df_seabed[['utm_x', 'utm_y']].values
    values = df_seabed['elevation'].values  
    # Perform Thin Plate Spline interpolation
    rbf = Rbf(points[:, 0], points[:, 1], values, function='thin_plate')
    water_depth = rbf(x, y)
    return np.minimum(0, water_depth)  

# Calculate monopile cost [€]
def monopile_cost_function(rated_power, water_depth, CO2_tax):
    a = 5*1000 # €/(MW*m) -old value: 21.3 * 1000
    b = 0.45*(10**6) #€/MW -old value: 137.5 * 1000
    monopile_cost = (a * water_depth-b) * rated_power  # Cost per turbine (negative since water depth is negative)
    monopile_mass = monopile_cost / 5.5 #5.5Eur/kg so mass is in kg
    #print(f"Monopile mass: {monopile_mass/1000} t")
    CO2e_monopile = (monopile_mass/1000) * 3.62 #3.62tCO2e/t steel so CO2e is in tCO2e
    monopile_tax = CO2e_monopile * CO2_tax # €/tCO2e
    monopile_cost = monopile_cost + monopile_tax # Add tax to the monopile cost
    return monopile_cost , monopile_tax

# Combine water depth and monopile cost functions
def combined_foundation_cost_function(x, y, site_name, rated_power, CO2_tax, seabed_data_path=None):
    water_depth = water_depth_func(x, y, site_name, seabed_data_path)
    monopile_cost, monopile_tax = monopile_cost_function(rated_power, water_depth, CO2_tax)
    return monopile_cost

def turbine_CO2_tax(wt, rated_power, CO2_tax, n_wt):
    
    power_kW = rated_power * 1000  # Convert to kW

    #Calculations for the tower mass, from INNWIND model
    tower_mass = (power_kW/10000)**(2.5/2)*628500*n_wt # kg

    #Calculations for the nacelle mass, from INNWIND model
    low_speed_shaft = (power_kW/5000)**(3/2)*27210
    main_bearing = 0.0092*(8/600*wt.diameter()-0.033)*(wt.diameter()**2.5)*2
    mechanical_brake_couplings = (power_kW/5000)**(3/2)*1000
    bed_plate = wt.diameter()**1.953 *1.295
    hydraulic_cooling_system = 0.08*power_kW
    nacelle_cover = 13000*(power_kW/5000)**(2/2)
    yaw_system = wt.diameter()**3.314 *0.0009*1.6
    omega = (0.0013*power_kW+76.904)/(wt.diameter()/2) #
    mu = 0.92
    tau = power_kW/(omega*mu)
    gearbox = 0.65*13.25*tau
    generator = 0.35*13.25*tau #kg
    nacelle_drivetrain_mass = (gearbox+generator+low_speed_shaft+main_bearing+mechanical_brake_couplings+bed_plate+hydraulic_cooling_system+nacelle_cover+yaw_system)*n_wt

    #Calculations for the blades mass, from INNWIND model
    blades_mass = 3*(power_kW/10000)**(3/2)*41716*n_wt # kg

    #Calculations for the hub+pitch mechanism mass, from INNWIND model
    hub_mass = (power_kW/5000)**(2.3/2)*40000*n_wt # kg
    pitch_mass = (power_kW/5000)**(2.3/2)*15000*n_wt # kg
    hub_pitch_mass = hub_mass + pitch_mass # kg

    #Material breakdown and CO2e emission
    steel_mass = tower_mass + nacelle_drivetrain_mass*0.885 + hub_pitch_mass*0.97 # kg
    CO2e_steel = (steel_mass/1000) * EF_steel #3.62tCO2e/t steel so CO2e is in tCO2e

    resin_mass = nacelle_drivetrain_mass*0.04 + blades_mass*0.265 + hub_pitch_mass*0.012
    CO2e_resin = (resin_mass/1000) * EF_resin

    glass_mass = nacelle_drivetrain_mass*0.04 + blades_mass*0.645 + hub_pitch_mass*0.012
    CO2e_glass = (glass_mass/1000) * EF_glass

    foam_mass = blades_mass*0.09
    CO2e_foam = (foam_mass/1000) * EF_foam
    
    Cu_mass = nacelle_drivetrain_mass*0.03
    CO2e_Cu = (Cu_mass/1000) * EF_Cu

    magnets_mass = nacelle_drivetrain_mass*0.005
    CO2e_magnets = (magnets_mass/1000) * EF_magnets

    cable_length = (200*10**-6)*power_kW*n_wt # m (200 km of cable for 1 GW)
    CO2e_cable = cable_length * EF_cable # tCO2e/km so CO2e is in tCO2e

    #Total CO2e emissions and tax
    total_turbine_tax = (CO2e_steel+CO2e_resin+CO2e_glass+CO2e_foam+CO2e_Cu+CO2e_magnets+CO2e_cable) * CO2_tax # €/tCO2e

    return total_turbine_tax


# Find LCOE CAPEX
def LCOE_CAPEX(AEP, monopile_cost, total_turbine_tax, **kwargs):
    
    AEP = AEP * 1000  # MWh  
    monopile_cost = np.abs(monopile_cost)  
    cost_CAPEX = cost_WT * n_wt * rated_power + monopile_cost + total_turbine_tax # € (Calculate cost of all capital expenditures)
    LCOE_CAPEX = ((cost_CAPEX)/(AEP))*CRF  # €/MWh (Calculate the LCOE CAPEX from the wind farm)
    return [LCOE_CAPEX, cost_CAPEX]

# Find LCOE OPEX
def LCOE_OPEX(AEP, **kwargs):
    AEP = AEP * 1000 # MWh
    print(AEP)
    LCOE_OPEX = (cost_fixed*n_wt*rated_power)/AEP + cost_var # €/MWh
    return LCOE_OPEX

# Find LCOE
def calculate_LCOE(LCOE_OPEX, LCOE_CAPEX, **kwargs):
    LCOE = LCOE_CAPEX + LCOE_OPEX # €/MWh
    return LCOE

#%% -------------------------------------------------------------------------------%%#
#                        DEFINE FOUNDATION COST MODEL COMPONENT                      #                #
#------------------------------------------------------------------------------------#
from topfarm.cost_models.cost_model_wrappers import CostModelComponent

class FoundationCostModelComponent(CostModelComponent):
    """Custom CostModelComponent to calculate foundation cost in euros."""

    def __init__(self, input_keys, n_wt, site_name, rated_power, CO2_tax, seabed_data_path=None, **kwargs):
        """Initialize the CustomCostModelComponent for foundation cost calculation."""
        super().__init__(input_keys, n_wt, **kwargs)  # Correctly call the parent class constructor
        self.site_name = site_name
        self.rated_power = rated_power
        self.seabed_data_path = seabed_data_path
        self.CO2_tax = CO2_tax

        def foundation_cost_function(**kwargs):
             """Calculate the foundation cost. The same as water_depth_func and monopile_cost_function - but this is needed to save the cost correctly in this class."""
             x = kwargs['x']
             y = kwargs['y']
             foundation_cost = combined_foundation_cost_function(x, y, self.site_name, self.rated_power,self.CO2_tax, self.seabed_data_path)
             return np.sum(abs(foundation_cost))  # Total cost for all turbines rounded to 2 decimals

        self.output_unit = 'M€'  # Set output unit to euros
        self.output_key = "Foundation Cost"
        self.cost_function = foundation_cost_function

        def foundation_cost_gradient(**kwargs):
            """Gradient of the foundation cost (optional, can be implemented if needed)."""
            raise NotImplementedError("Gradient calculation is not implemented for foundation cost.")

        self.cost_gradient_function = foundation_cost_gradient

    def get_cost4smart_start(self, **kwargs):
        """Compute foundation cost for smart start."""
        def cost4smart_start(X, Y, wt_x, wt_y, **kwargs):
            foundation_cost = combined_foundation_cost_function(X.flatten(), Y.flatten(), self.site_name, self.rated_power, self.CO2_tax, self.seabed_data_path)
            return foundation_cost.reshape(X.shape)

        return cost4smart_start

#%%
def get_problem(wt, EIB_boundary, wf_model, n_wt, site_name, rated_power, n_spacing, CO2_tax, total_turbine_tax):

    # ------------------------------ Define spacing constraint -----------------------------#
    spacing_constraint = n_spacing * wt.diameter()  # Minimum distance between wind turbines

    # ----------------- Define the initial positions of the wind turbines ------------------#
    x_init = np.random.uniform(EIB_boundary[:, 0].min(), EIB_boundary[:, 0].max(), size=n_wt)
    y_init = np.random.uniform(EIB_boundary[:, 1].min(), EIB_boundary[:, 1].max(), size=n_wt)

    # ----------------------------------- Cost components ----------------------------------#
    aep_comp = PyWakeAEPCostModelComponent(wf_model, n_wt, grad_method=autograd,objective=False, output_keys=[('AEP',0)])   

    foundation_comp = FoundationCostModelComponent(input_keys=['x','y'], 
                                                n_wt=n_wt,
                                                rated_power=rated_power,
                                                site_name=site_name,
                                                CO2_tax=CO2_tax,
                                                cost_function=lambda x, y: FoundationCostModelComponent.foundation_cost_function(x, y, site_name, rated_power, CO2_tax),
                                                objective=False,
                                                output_keys=[('monopile_cost', 0)])

    LCOE_CAPEX_comp = CostModelComponent(input_keys=[('AEP', 0), ('monopile_cost', 0)],
                                          n_wt=n_wt,
                                          cost_function=lambda AEP, monopile_cost: LCOE_CAPEX(AEP, monopile_cost, total_turbine_tax=total_turbine_tax),
                                          objective=False,
                                          output_keys=[('LCOE_CAPEX', 0), ('cost_CAPEX', 0)])


    LCOE_OPEX_comp = CostModelComponent(input_keys=[('AEP', 0)],
                                          n_wt=n_wt,
                                          cost_function=LCOE_OPEX,
                                          objective=False,
                                          output_keys=[('LCOE_OPEX',0)])
    
    LCOE_comp = CostModelComponent(input_keys=[('LCOE_CAPEX', 0), ('LCOE_OPEX', 0)],
                              n_wt=n_wt,
                              cost_function=calculate_LCOE,
                              objective=True,
                              maximize=False,
                              output_keys=[('LCOE', 0)], output_unit='€/MWh')



    # Create a TopFarmGroup and add subsystems
    cost_components = TopFarmGroup([aep_comp, foundation_comp, LCOE_CAPEX_comp, LCOE_OPEX_comp, LCOE_comp])

    #---------------------------- Define the optimization problem--------------------#
    problem = TopFarmProblem(
        design_vars={'x': x_init, 'y': y_init},
        driver=EasyScipyOptimizeDriver(optimizer='SLSQP', disp=False),
        cost_comp=cost_components,
        constraints=[
            XYBoundaryConstraint(np.array([EIB_boundary[:, 0], EIB_boundary[:, 1]]).T, boundary_type='polygon'),
            SpacingConstraint(spacing_constraint)
        ],
        plot_comp=XYPlotComp()
    )
    
    return problem, aep_comp, foundation_comp

#%% -------------------------------------------------------------------------------%%#
#                                 SMART START                                        #
#------------------------------------------------------------------------------------#
# Initialize an empty DataFrame to store results
results_df = pd.DataFrame(columns=['turbine [MW]', 'site', 'seed', 'LCOE [€/MWh]', 'avg_water_depth [m]', 'AEP [GWh]', 'AEP without wake loss [GWh]', 'Wake loss [%]'])

# Define turbines and sites to iterate over
turbines = [8, 10, 12, 15, 17, 20, 22]  # Turbine capacities in MW
sites = ['EIB1N','EIB1S', 'EIB2']  # Site names
seeds = range(1,11)    # Seeds   

CO2_tax = CO2_tax # €/tCO2e
n_spacing = 4

# Nested loops for turbines, sites, and seeds
for turbine in turbines:
    wt, rated_power = set_wt(turbine, use_ref=False)  # Set turbine
    if wt is None:  # Skip if turbine is not defined
        continue
    
    for site_name in sites:
        # Get site-specific data
        EIB_boundary, x_site, y_site, site, n_wt, x_points, y_points, XX, YY = get_site(site_name, wt, rated_power)
        
        total_turbine_tax = turbine_CO2_tax(wt, rated_power, CO2_tax, n_wt) # Calculate the total tax for all turbines

        # Set wake model
        wf_model = set_wake_model('NOJ', site, wt)  # Using NOJ as the wake model

        # Define the optimization problem
        problem, aep_comp, foundation_comp = get_problem(wt, EIB_boundary, wf_model, n_wt, site_name, rated_power, n_spacing, CO2_tax, total_turbine_tax)
        
        for seed in seeds:

            aep_func = aep_comp.get_aep4smart_start()
            cost_func = foundation_comp.get_cost4smart_start()

            def zz_func(X, Y, wt_x, wt_y, *args, **kwargs):
                # Calculate AEP and cost
                aep = aep_func(X, Y, wt_x, wt_y, *args, **kwargs)  # AEP in GWh
                monopile_cost = cost_func(X, Y, wt_x, wt_y, *args, **kwargs)  # Cost in M€

                # Convert AEP to MWh for LCOE calculation
                aep_mwh = aep*1000   # Convert GWh to MWh

                # Calculate CAPEX and OPEX components
                capex = -cost_WT *  rated_power + monopile_cost - total_turbine_tax # Total CAPEX in M€
                opex = -(cost_fixed *  rated_power) - (cost_var * aep_mwh)  # Total OPEX in €/year

                # Calculate LCOE (€/MWh)
                lcoe = ((capex * CRF) + opex) / aep_mwh if np.any(aep_mwh > 0) else np.inf  # Avoid division by zero
                return lcoe
            
            # Perform smart start optimization
            problem.smart_start(
                XX,
                YY,
                ZZ=zz_func,
                min_space=n_spacing * wt.diameter(),  # Spacing constraint
                random_pct=0.8,
                seed=seed
            )
            
            # Evaluate the problem
            cost_smart, state_smart = problem.evaluate()
            sim_res = wf_model(state_smart['x'], state_smart['y'])
            aep_with_wake_loss = sim_res.aep(with_wake_loss=True).sum().data
            aep_without_wake_loss = sim_res.aep(with_wake_loss=False).sum().data
            # average water depth of turbines
            water_depth = water_depth_func(state_smart['x'], state_smart['y'], site_name)
            avg_water_depth = np.mean(water_depth)
            total_monopile_cost, total_monopile_tax = n_wt*np.abs(monopile_cost_function(rated_power, avg_water_depth, CO2_tax))*10**-6 # Cost in M€
            # Append the results to the DataFrame
            results_df = pd.concat([results_df, pd.DataFrame({
                'turbine [MW]': [turbine],
                'site': [site_name],
                'seed': [seed],
                'LCOE [€/MWh]': [abs(cost_smart)],
                'avg_water_depth [m]': [avg_water_depth],
                'AEP [GWh]': [aep_with_wake_loss],
                'AEP without wake loss [GWh]': [aep_without_wake_loss],
                'Wake loss [%]': [((aep_without_wake_loss - aep_with_wake_loss)/aep_without_wake_loss) * 100],
                'Monopile Cost [M€]': [total_monopile_cost],
                'Monopile Tax [M€]': [total_monopile_tax],
                'CAPEX [M€]': [cost_WT*10**-6 * n_wt * rated_power + total_monopile_cost + total_turbine_tax*10**-6],
                'Turbine Tax [M€]': [total_turbine_tax*10**-6],
            })], ignore_index=True)
    print()
    print("----------------------------------------------")
    print(f"Completed turbine {turbine}MW")
    print("----------------------------------------------")
    print()

# Print the resulting DataFrame
print(results_df)

#%%---------------------------------------------------------------------------%%#
#                          PLOTTING FUNCTION (seabed + smartstart)                               #
#-------------------------------------------------------------------------------#
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
    cbar.set_label('Seabed Elevation [m]',fontsize=16)

    # Plot boundary as a polygon
    ax.plot(df_boundary['X'], df_boundary['Y'], color='black', linestyle='-', linewidth=1.5, label='Boundary')
    # Close the polygon by connecting the last point to the first
    ax.plot([df_boundary['X'].iloc[-1], df_boundary['X'].iloc[0]], 
            [df_boundary['Y'].iloc[-1], df_boundary['Y'].iloc[0]], 
            color='black', linestyle='-', linewidth=1.5)

    # Plot turbine positions
    ax.scatter(state_smart['x'], state_smart['y'], c='red', label=f'Turbine Positions', marker='x')

    # Set legend in the lower left corner with font size
    ax.legend(loc='upper left', prop={'size': 19}, framealpha=0.7)  # Position legend in the lower right corner with transparency

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
    ax.set_xlabel('UTM Easting [m]',fontsize=16)
    ax.set_ylabel('UTM Northing [m]',fontsize=16)
    ax.locator_params(axis='x', nbins=6)
    ax.tick_params(axis='both', which='major', labelsize=16)
    cost_text = f"{cost:.1f}" if cost is not None else "N/A"
    ax.set_title(f'{turbine}MW Turbines, {site_name} \n {cost_key}: {cost_text} {output_unit}' ,fontsize=18) # Turbine Positions and Seabed Data for {site_name} \n
    ax.axis('equal')  # Ensure equal axis scaling
    plt.show()

plot_state_with_seabed(state_smart, site_name=site_name, turbine=turbine, seabed_data_path=None, output_unit='€/MWh', cost=cost_smart, cost_key='LCOE', use_ref=False)



#%%----------------------------------------------------------------------------%%#
#                      Save the results to a CSV file                            
#--------------------------------------------------------------------------------#

# # CAREFULL NOT TO OVERWRITE THE FILE
#results_df.to_csv('Smart_start_LCOE_baseline_biggerturbines2530', index=False)
# # CAREFULL NOT TO OVERWRITE THE FILE


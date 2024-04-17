#Load packages
import pulp
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

class Battery():

    def __init__(self, 
                 time_horizon,
                 max_disharge_power_capacity,
                 max_charge_power_capacity):
        #set up decision variables for optimization.
        #Hourly cahrge and discharge flows with limitations
        #Optimization horizon, hours
        self.time_horizon = time_horizon

        self.charge = \
        pulp.LpVariable.dicts(
            "charging power",
            ('c_t_' + str(i) for i in range(0, time_horizon)),
            lowBound=0, upBound=max_charge_power_capacity,
            cat='Continuous')
        
        self.discharge = \
        pulp.LpVariable.dicts(
            "discharging power",
            ('d_t_' + str(i) for i in range(0, time_horizon)),
            lowBound=0, upBound=max_disharge_power_capacity,
            cat='Continuous')
        
    def set_objective(self, prices):
        #create a model and objective funciton.
        #using price data, one price per point on horizon.
        #future implementation; add price likelihoods > Expected Value
        try:
            assert len(prices) == self.time_horizon
        except:
            print('Error: need one price for each hour in time horizon')

            #Instantiate Linear Programming model to maximize objective
            self.model = pulp.LpProblem("Energy arbitrage",
                                        pulp.LpMaximize)
            
            #OBJECTIVE FUNCTION: PROFIT
            #Daily Profit from charging/discharging. 
            #Charging as cost, discharging as revenue.
            self.model += \
            pulp.LpAffineExpression(
                [(self.charge['c_t' + str(i)],
                  -1*prices[i]) for i in range(0, self.time_horizon)]) +\
            pulp.LpAffineExpression(
                [(self.discharge['d_t' + str(i)],
                  prices[i]) for i in range(0, self.time_horizon)])
            
    def add_storage_contraints(self,
                               efficiency,
                               min_capacity,
                               max_capacity,
                               initial_level):
        #minimum storage level constraint
        #round trip efficiency: energy available for disharge x energy charged
        for hour_of_sim in range(1, self.time_horizon+1):
            self.model += \
            initial_level \
            + pulp.LpAffineExpression(
                [(self.charge['c_t_' + str(i)], efficiency)
                 for i in range(0, hour_of_sim)]) \
            - pulp.lpSum(
                self.discharge[index]
                for index in('d_t_' + str(i)
                             for i in range(0, hour_of_sim)))\
            >= min_capacity

        #Storage Level Constraint 2
        #Max Energy Capacity - Min Capacity = Discharge Energy Capacity
        for hour_of_sim in range(1, self.time_horizon+1):
            self.model += \
            initial_level \
            + pulp.LpAffineExpression(
                [(self.charge['c_t_' + str(i)], efficiency)
                 for i in range(0, hour_of_sim)]) \
            - pulp.lpSum(
                self.discharge[index]
                for index in('d_t_' + str(i)
                             for i in range(0, hour_of_sim)))\
            <= max_capacity

def add_throughput_constraints(self,
                               max_daily_discharged_throughput):
    #Max Discharge Throughput Constraint
    #The sum of all discharge flow within a day cannot exceed this
    #Include portion of the next day according to time horizon
    #Assumes the time horizon is at least 24 hours
    #Should be replaced with a discharge cost function, see battery opt research
    self.model += \
    pulp.lpSum(
        self.discharge[index] for index in (
            'd_t_' + str(i) for i in range(0, 24))) \
        <= max_daily_discharged_throughput

    self.model += \
    pulp.lpSum(
        self.dischacharge[index] for index in range(25, self.time_horizon)))\
        <= max_daily_discharged_throughput \
        *float(self.time_horizon-24)/24

def solve_model(self):
    #solve optimization problem, subject to constraints
    self.model.solve()

    #show warning if optimal solution not found
    if pulp.LpStatus[self.model.status] != 'Optimal':
        print('Warning:' + pulp.LpStatus[self.model.status])

def collect_output(self):
    #collect charging and discharging rates within time horizon
    hourly_charges =\
        np.array(
            [self.cahrge[index].varValue for 
             index in ('c_t_' + str(i) for i in range(0, 24))])
    hourly_discharges =\
        np.array(
            [self.discharge[index].varValue for
             index in ('d_t_' + str(i) for i in range(0, 24))])
    return hourly_charges, hourly_discharges


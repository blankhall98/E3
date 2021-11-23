import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pprint
import inspect

###################  
class SDG:
    
    def __init__(self,inputs):
        
        self.inputs = inputs
        self.goal = inputs.goal
        self.formula = inputs.formulas[str(self.goal)]
    
        #DataFrames
        self.GDPpp = pd.read_csv(inputs.GDPpp_source,index_col='Region').apply(pd.to_numeric)
        self.population = pd.read_csv(inputs.population_source,index_col='Region').apply(pd.to_numeric)
        self.sdg = self.load_sdg_information()
        self.threshold = self.load_threshold_information()
        
        #Correlation
        self.correlation_parameters = {}
        
        #Performance
        self.regional_performance = None
        self.world_performance = None
        
    def __str__(self):
        goal_name = self.threshold['goal_name'].values[0]
        indicator = self.threshold['indicator'].values[0]
        return str(f'{goal_name}: {indicator}')
        
    #Loads the SDG data for the specific goal
    def load_sdg_information(self):
        db = pd.read_csv(self.inputs.sdg_source,index_col='Region')
        db['SDG'] = db['SDG'].astype(int)
        db = db[db['SDG'] == self.inputs.goal]
        db = db[[str(x) for x in self.inputs.total_time]]
        return db
    
    #Loads the threshold information for specific goal
    def load_threshold_information(self):
        db = pd.read_csv(self.inputs.threshold_source,index_col='goal')
        db = db[db.index == self.goal]
        return db
        
    ##### COMPUTE #####
    
    # Makes pertinent computations to complete the model
    def compute(self):
        self.correlate()
        self.predict()
        self.compute_regional_performance()
        self.compute_world_performance()
    
    ### CORRELATE ###
    
    # Depending on the goal, selects a correlation methodology. Loads information about correlation parameters
    def correlate(self):
        if self.goal == 1:
            self.correlate_sdg1()
        elif self.goal == 2:
            self.correlate_sdg2()
                
    #SDG1 correlation methodology
    def correlate_sdg1(self):
        for region in self.sdg.index:
            params = []
            if region == 'China':
                params.append(140)
            else:
                params.append(100)
                
            formula = lambda x,b : params[0]*np.exp(-x/b)
        
            region_sdg = self.sdg.loc[region][[str(x) for x in self.inputs.historic_time]].values
            region_gdp = self.GDPpp.loc[region][[str(x) for x in self.inputs.historic_time]].values
            
            parameters, covariance = curve_fit(formula,region_gdp,region_sdg)
            for x in parameters:
                params.append(x)
            
            self.correlation_parameters[region] = {
                'parameters': params,
                'covariance': covariance[0][0]
                }
            
    def correlate_sdg2(self):
        for region in self.sdg.index:
            params = []
            formula = lambda x,b: 2.5 + 32.5*np.exp(-x/b)
        
            region_sdg = self.sdg.loc[region][[str(x) for x in self.inputs.historic_time]].values
            region_gdp = self.GDPpp.loc[region][[str(x) for x in self.inputs.historic_time]].values
        
            parameters, covariance = curve_fit(formula,region_gdp,region_sdg)
            
            for x in parameters:
                params.append(x)
            
            self.correlation_parameters[region] = {
                'parameters': params,
                'covariance': covariance[0][0]
                }
        
            
    
    ### PREDICT ###
    
    #Computes and loads predictions of the indicator based on the fitted correlation
    def predict(self):
        for region in self.sdg.index:
            region_gdp = self.GDPpp.loc[region][[str(x) for x in self.inputs.prediction_time]].values
            region_pred = [self.formula(x,self.correlation_parameters[region]['parameters']) for x in region_gdp]
            self.sdg.loc[region][[str(x) for x in self.inputs.prediction_time]] = region_pred
            
    
    ##### PLOT #####
    
    #Plots the Indicator's historic time series
    def plot_historic(self):
        time = self.inputs.historic_time
        plt.figure(figsize = (12,8))
        plt.title(f'Historic Data of {self.__str__()}')
        for region in self.sdg.index:
            region_name = region
            region_data = self.sdg.loc[region][[str(x) for x in time]].values
            plt.plot(time,region_data,label=region_name)
        plt.xlabel('Date')
        plt.ylabel(f'{self.__str__()}')
        plt.legend()
        plt.show()
        
    def plot_prediction(self):
        time = self.inputs.prediction_time
        plt.figure(figsize = (12,8))
        plt.title(f'Prediction of {self.__str__()}')
        for region in self.sdg.index:
            region_name = region
            region_data = self.sdg.loc[region][[str(x) for x in time]].values
            plt.plot(time,region_data,label=region_name)
        plt.xlabel('Date')
        plt.ylabel(f'{self.__str__()}')
        plt.legend()
        plt.show()
    
    def plot_total(self):
        time = self.inputs.total_time
        green2yellow = self.threshold['green-to-yellow']
        yellow2red = self.threshold['yellow-to-red']
        plt.figure(figsize = (12,10))
        plt.title(f'Historic and Prediction Data of {self.__str__()}')
        for region in self.sdg.index:
            region_name = region
            region_data = self.sdg.loc[region][[str(x) for x in time]].values
            plt.plot(time,region_data,label=region_name)
        plt.hlines(green2yellow,min(time),max(time),linestyles = "dotted", colors= "green",label="Green to Yellow Threshold")
        plt.hlines(yellow2red,min(time),max(time),linestyles = "dotted", color = "red",label="Yellow to Red Threshold")
        plt.xlabel('Date')
        plt.ylabel(f'{self.__str__()}')
        plt.legend()
        plt.show()
    
    #Plots the correlation between GDPpp and SDG
    def plot_correlation(self):
        plt.figure(figsize = (12,8))
        plt.title(f'Correlation between GDP per Capita and {self.__str__()}')
        for region in self.sdg.index:
            region_name = region
            region_sdg = self.sdg.loc[region][[str(x) for x in self.inputs.historic_time]].values
            region_gdp = self.GDPpp.loc[region][[str(x) for x in self.inputs.historic_time]].values
            plt.plot(region_gdp,region_sdg,'o',label=region_name)
        plt.xlabel('GDP per Capita')
        plt.ylabel(f'{self.__str__()}')
        plt.legend()
        plt.show()
    
    #Plots the fitted correlation between GDPpp and SDG
    def plot_fit(self):
        gdp_domain = np.arange(0,80,1)
        plt.figure(figsize=(12,8))
        plt.title(f'Correlation fit between GDP per Capita and {self.__str__()}')
        for region in self.correlation_parameters.keys():
            region_name = region
            indicator = [self.formula(x,self.correlation_parameters[region]['parameters']) for x in gdp_domain]
            plt.plot(gdp_domain,indicator,label=region_name)
        plt.xlabel('GDP per Capita')
        plt.ylabel(f'{self.__str__()}')
        plt.legend()
        plt.show()
    
    def plot_performance(self):
        time = self.inputs.total_time
        performance = self.world_performance.values
        plt.figure(figsize=(12,8))
        plt.title(f'Worlds perfromance at {self.__str__()}')
        plt.xlabel('Date')
        plt.ylabel('Worlds Performance')
        plt.plot(time,performance,label=f'Goal {self.goal}')
        plt.legend()
        plt.show()
    
    #Plot all the method's plots
    def display(self):
        self.plot_historic()
        self.plot_correlation()
        self.plot_fit()
        self.plot_prediction()
        self.plot_total()
        self.plot_performance()
        
    ##### PERFORMANCE #####

    def compute_regional_performance(self):
        rp = pd.DataFrame(index = self.sdg.index.to_list(), columns=self.inputs.total_time)
        sign = self.threshold["sign"].values[0]
        green_yellow = self.threshold["green-to-yellow"].values[0]
        yellow_red = self.threshold["yellow-to-red"].values[0]
        
        for region in rp.index:
            region_indicator =  self.sdg.loc[region].values
            region_grade = []
            if sign == "less_than":
                for value in region_indicator:
                    if value < green_yellow:
                        region_grade.append(1)
                    elif value < yellow_red:
                        region_grade.append(0.5)
                    else:
                        region_grade.append(0)
            elif sign == "more_than":
                for value in region_indicator:
                    if value > green_yellow:
                        region_grade.append(1)
                    elif value > yellow_red:
                        region_grade.append(0.5)
                    else:
                        region_grade.append(0)
            
            rp.loc[region] = region_grade
            
        self.regional_performance = rp.copy()
    
    def compute_world_performance(self):
        population_proportion = self.population/self.population.sum()
        self.world_performance = pd.DataFrame(self.regional_performance.values*population_proportion.values,
                                              index=self.regional_performance.index,
                                              columns=self.regional_performance.columns)
        self.world_performance = self.world_performance.sum()
        
        
    ##### REPORT #####
    def report(self):
        pp = pprint.PrettyPrinter(indent=2)
        print('\n' + f'SDG{self.goal} --- {self.__str__()}')
        
        print('\n' + "Worlds Performance:")
        print(self.world_performance)
        
        print('\n' + "Regional Performance:")
        print(self.regional_performance)
        
        print('\n' + "Regression Formula:")
        print(inspect.getsource(self.formula))
        
        print('\n' + "Regression Coefficients:")
        pp.pprint(self.correlation_parameters)
        
            
        

###################     
class sdg_inputs:
    
    def __init__(self):
        self.goal = None
        self.historic_time = np.arange(1980,2020,5,dtype=int)
        self.prediction_time = np.arange(2020,2055,5,dtype=int)
        self.total_time = np.arange(1980,2055,5,dtype=int)
        
        #Data Sources
        self.GDPpp_source = './data/GDPpp.csv'
        self.population_source = './data/Population.csv'
        self.sdg_source = './data/SDG1-7.csv'
        self.threshold_source = './data/Threshold.csv'
        
        #Formulas
        self.formulas = {
            '1': lambda x,params: params[0]*np.exp(-x/params[1]),
            '2': lambda x,params: 2.5 + 32.5*np.exp(-x/params[0]),
            '3': None,
            '4': None,
            '5': None,
            '6': None,
            '7': None
            }

        
        
        
        
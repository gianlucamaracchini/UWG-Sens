import numpy as np
from uwg import *
from uwg import Element, Material, Building, BEMDef, SchDef
import pandas as pd
import multiprocessing as mp
import time
import os
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.analyze import morris as M
from SALib.analyze.morris import compute_elementary_effects
from SALib.sample.morris import _compute_delta
import matplotlib.pyplot as plt
import shutil
import parmap
from os import path
from pvlib import iotools
import sys

data = pd.DataFrame()
Temp = pd.DataFrame()

#%%

project_folder = os.path.dirname(__file__) 
results = os.path.join(project_folder, "results")
resources = os.path.join(project_folder, "resources")
parameters = os.path.join(resources, "parameters")
epw_folder = os.path.join(resources, "epw")
num_CPUs = mp.cpu_count()-1
epw_name = "EPW.epw"

#%%
epw = os.path.join(epw_folder, epw_name)

if __name__ == '__main__':
    if path.exists(epw) == False:
        print(".epw file not found.")
        sys.exit()
    else:
        pass
    
if __name__ == '__main__':
    try:
        os.mkdir(results)
    except OSError:
        try:
            os.mkdir(os.path.join(project_folder, "results_old")) #new directory created
        except OSError:
            shutil.rmtree(os.path.join(project_folder, "results_old")) # deleting existing result_old folder and related file
            os.mkdir(os.path.join(project_folder, "results_old")) # creating empty result_old folder
            for file in os.listdir(results):
                shutil.move(os.path.join(results, file), os.path.join(project_folder, "results_old")) # and old files moved to new result_old folder
        print (" Existing 'results' subfolder has been renamed 'results_old'. The directory results has been successfully created" )
    else:
        os.mkdir(results)
        print ("Successfully created the directory %s " % results)

#%%
import pandas as pd
import json
if __name__ == '__main__':
    if input("Do You Want To Carry Out Single Simulation? [y/n] ") == "n":
        pass
    else:
        #from uwg.cli.simulate import simulate_json_model
        model_json = os.path.join(project_folder,"custom_uwg.json")
    
        with open(model_json) as json_file:
            data = json.load(json_file)
    
        model = UWG.from_dict(data, epw, new_epw_dir=results)
        model.epw_precision = 3
        model.generate()       
        model.simulate()
        model.write_epw()
        output = pd.DataFrame()
        
        CanTemp = []
        UBLTemp = []
        RoofTemp = []
        #CeilTemp = []
        WallTemp = []
        UCM_Qubl = []
        UCM_Qhvac = []
        UCM_Qroof = []
        UCM_Qwall = []
        #ExtSurfRoofFlux = []
        #IntRoofFlux = []
        #indoorTemp = []
        for j in range(len(model.UCMData)):
            CanTemp.append(model.UCMData[j].canTemp - 273.15)
            UBLTemp.append(model.UBLData[j].ublTemp - 273.15)
            RoofTemp.append(model.UCMData[j].roofTemp - 273.15) # Average (among building typologies) roof temperature
            #CeilTemp.append(model.UCMData[j].ceilTemp - 273.15) # Average (among building typologies) ceil temperature
            WallTemp.append(model.UCMData[j].wallTemp - 273.15) # Average wall temperature
            UCM_Qubl.append(model.UCMData[j].Q_ubl) # Convective heat exchange with UBL layer
            UCM_Qhvac.append(model.UCMData[j].Q_hvac) # Sensible heat flux from HVAC waste
            UCM_Qroof.append(model.UCMData[j].Q_roof) # Sensible heat flux from building roof (convective)
            UCM_Qwall.append(model.UCMData[j].Q_wall) # Sensible heat flux from building wall (convective)
            #indoorTemp.append(model.UCMData[j].indoorTemp - 273.15) #modificato UCMDef.py per estrapolare i dati interni
            #ExtSurfRoofFlux.append(model.UCMData[j].ExtSurfRoofFlux) 
            #IntRoofFlux.append(model.UCMData[j].IntRoofFlux) #positivo se Tindoor>Tceil (negativo se entrante)
                
        output['CanTemp'] = CanTemp
        output['UBLtemp'] = UBLTemp
        output['RoofTemp'] = RoofTemp
        #output['CeilTemp'] = CeilTemp
        output['WallTemp'] = WallTemp
        output['UCM_Qubl'] = UCM_Qubl
        output['UCM_Qhvac'] = UCM_Qhvac
        output['UCM_Qroof'] = UCM_Qroof
        output['UCM_Qwall'] = UCM_Qwall
        #output['indoorTemp'] = indoorTemp
        #output['ExtSurfRoofFlux'] = ExtSurfRoofFlux
        #output['IntRoofFlux'] = IntRoofFlux
        
        output.to_excel(os.path.join(results, "output.xlsx"))
       
#%%
def names_and_bounds(name, value, names, bounds):
    if type(value) is str:
        if value.startswith("@"):
            print(name)
            print(value + "\n")
            names.append(name)
            minmax = value.replace("@","")
            bound1 = list(minmax.split(","))
            bound2 = list(map(float, bound1))
            bounds.append(bound2) 
    return names, bounds

def json_parametrizing(dict_data, variables_names, variables_bounds, base_name=""):
    
    for key, value in dict_data.items():
               
        #string at the main level
        if type(value) is str:
            name = base_name + key
            variables_names, variables_bounds = names_and_bounds(name, value, variables_names, variables_bounds)
        
        #list at the main level
        elif type(value) is list:
                      
            k = 0           
            for value1 in value:      
                name = base_name + key + "_" + str(k)
                
                #list with strings                       
                if type(value1) is str:
                    variables_names, variables_bounds = names_and_bounds(name, value1, variables_names, variables_bounds)
                    break
                
                #list of lists    
                elif type(value1) is list:
                    j = 0
                    for value2 in value1: 
                        #list of lists with strings                                                                       
                        if type(value2) is str:
                            variables_names, variables_bounds = names_and_bounds(name + "_" + str(j), value2, variables_names, variables_bounds) 
                        j += 1                                                  
                
                #list of dicts
                elif type(value1) is dict:
                    variables_names, variables_bounds = json_parametrizing(value1, variables_names, variables_bounds, base_name=name + "_")                   
                
                k += 1
        
        #dict at the main level
        elif type(value) is dict:
            name = base_name + "_" + key
            variables_names, variables_bounds = json_parametrizing(value, variables_names, variables_bounds, base_name=name + "_")
                
    #print(variables_names, variables_bounds)
    return variables_names, variables_bounds

#%%
def data_input1(i, data, X, k=0):    
    for key, value in data.items():       
        #k=0
        #string at the main level
        if type(value) is str:
            if value.startswith("@"):
                data[key] = float(X[i,k])
                k += 1
        
        #list at the main level
        elif type(value) is list:                 
            for j in range(len(value)):  
                #list with strings                       
                if type(value[j]) is str:
                    if value[j].startswith("@"):
                        data[key][j] = float(X[i,k] )    
                        k += 1
                
                #list of lists    
                elif type(value[j]) is list:
                    for y in range(len(value[j])):                      
                        #list of lists with strings                                                                       
                        if type(value[j][y]) is str:
                            if value[j][y].startswith("@"):
                                data[key][j][y] = float(X[i,k]) 
                                k += 1                                
                                                             
                #list of dicts
                elif type(value[j]) is dict:
                    data[key][j], k = data_input1(i, value[j], X, k)   
        
        #dict at the main level
        elif type(value) is dict:
            data[key], k = data_input1(i, value, X, k)
               
    return data, k
                    
def key_value(data, key, value):
    
    for key1, value1 in data.items():
        #string at the main level
        if value1 == key:
            data[key1] = value 
        
        #list at the main level
        elif type(value1) is list:                 
            for j in range(len(value1)):  
                #list with strings                       
                if value1[j] == key:
                    data[key1][j] = value 
                
                #list of lists    
                elif type(value1[j]) is list:
                    for y in range(len(value1[j])):                      
                        #list of lists with strings                                                                       
                        if value1[j][y] == key:
                            data[key1][j][y] = value                                
                                                             
                #list of dicts
                elif type(value1[j]) is dict:
                    data[key1][j] = key_value(value1[j], key, value)   
        
        #dict at the main level
        elif type(value1) is dict:
            data[key1] = key_value(value1, key, value)
    
    return data

def multiplying(data):
    
    for key, value in data.items():
        if key.startswith("*"):
            string = key.replace("*", "")
            for key1, value1 in data.items():
                if key1 == string:
                    data[key1] = data[key] * data[key1]
        else:
            pass
        
        if type(value) is list:
            for i in range(len(value)):
                
                if type(value[i]) is float or int:
                    pass
                
                if type(value[i]) is list:
                    pass
                
                if type(value[i]) is dict:
                    data[key][i] = multiplying(value[i])

        elif type(value) is dict:
            data[key] = multiplying(value)
            
    return data

def complementary(data):

    for key, value in data.items():
        if type(value) is list:

            for i in range(len(value)):
                
                if type(value[i]) is list: 
                    for j in range(len(value[i])):
                        if type(value[i][j]) is str:                                                                     
                            if value[i][j].startswith("-"):
                                string = value[i][j].replace("-", "_")
                                for key1, value1 in data.items():
                                    if key1 == string:
                                        data[key][i][j] = 1 - data[string]
                
                elif type(value[i]) is dict:
                    data[key][i] = complementary(value[i])

        elif type(value) is dict:
            data[key] = complementary(value)
            
    return data
     
def data_input2(data):    

    for key, value in data.items():  
        if key.startswith("_"):
            new_data = key_value(data, key, value)

    new_data = multiplying(new_data)
    new_data = complementary(new_data)
               
    return new_data

def data_input3(data):  
    if data["#LCZ"]:
        LCZdata = data["#LCZ"]
        if LCZdata["translate"] == True:
            data["blddensity"] = LCZdata["bld_sf"]/100
            data["bldheight"] = LCZdata["meanHeight"]
            data["vertohor"] = 4 * (data["blddensity"]**0.5 - data["blddensity"]) * LCZdata["aspectratio"]

            data["treecover"] = LCZdata["perv_sf"]/100 * LCZdata["perc_treecover"]
            data["grasscover"] = LCZdata["perv_sf"]/100 - data["treecover"]
                
#            if LCZdata["perv_sf"] + LCZdata["bld_sf"] >= 100:
#                data["treecover"] = (1 - data["blddensity"]) * LCZdata["perc_treecover"]
#                data["grasscover"] = (1 - data["blddensity"]) - data["treecover"]
#            else:
#                data["treecover"] = LCZdata["perv_sf"]/100 * LCZdata["perc_treecover"]
#                data["grasscover"] = LCZdata["perv_sf"]/100 - data["treecover"]
            
            #print((data["treecover"]+data["grasscover"])/ (1 - data["blddensity"]))
    return data
       
#%%
if __name__ == '__main__':
    if input("Do You Want To Carry Out Sensitivity Analysis? [y/n] ") == "n":
        sys.exit()
    else:
        # read in the file
        variables_names = []
        variables_bounds = []
        model_json = os.path.join(project_folder,"custom_uwg_param.json")
        with open(model_json) as json_file:
            data = json.load(json_file)
        variables_names, variables_bounds = json_parametrizing(data, variables_names, variables_bounds)
        
        variables_values = pd.DataFrame(index=variables_names)
        ## Define the model inputs for sensitivity
        D = len(variables_names)
        problem = {'num_vars': D,
                   'names': variables_names,
                   'bounds': variables_bounds
                   }  
        N = data["#N"]
            
        if data["#sensitivity"] == 'Sobol':
            #n_sim = N * ( 2*D + 2 ) # == len(X), Num of simulations/output --> N(D+2) for first order sensitivity indexes, N(2D+2) for total order.
            X = saltelli.sample(problem, N, calc_second_order = data["#calc_second_order"], skip_values = data["#skipvalues"]) #-->  creates the Saltelli's matrixes and defines N
                
        elif data["#sensitivity"] == 'Morris':
            from SALib.sample import morris
            #n_sim = N * ( m + 1 )      # Num of model evaluation according to Morris.
            X = morris.sample(problem, N, num_levels = 4, seed=None)  

        for i in range(len(X)):
            variables_values[i] = np.asarray(X[i])
            np.transpose(variables_values)
        
#%% 
if __name__ == '__main__':        
    IO_matrix = pd.DataFrame(variables_values).T # creation of input-ouput matrix with input values
    IO_matrix.to_excel(os.path.join(results, "I-O_matrix.xlsx"), 
                          index = True, encoding = 'utf-8-sig')

#%%

def UHII(T_rur, T_urb, startmonth, startday, nday, warmingdays):

    uhii = []
    ucii = []
    cddd = []
    sumuhii = 0
    sumucii = 0
    hours = 0 + warmingdays*24
    m = startmonth
    sd = startday + warmingdays
    while hours/24 < nday:
        if T_urb[hours] > T_rur[hours]:
            uhii.append(T_urb[hours] - T_rur[hours])
            sumuhii = sumuhii + uhii[-1]
            ucii.append(0)
        else:
            uhii.append(0)
            ucii.append(T_rur[hours] - T_urb[hours])
            sumucii = sumucii + ucii[-1]
        cddd.append(T_urb[hours] - T_rur[hours])
        hours += 1
        if hours/24 == nday:
            break
        m += 1
        sd = 1
    return sumuhii/(hours-warmingdays*24), uhii, sum(cddd), sumucii/(hours-warmingdays*24), ucii

import statistics as st
def TmaxD(T_rur, T_urb, startmonth, startday, nday,  warmingdays):
    TmaxD = []
    days = 0
    m = startmonth
    sd = startday
    while days < nday:
        for d in range(int(len(T_urb)/24)):
            TDmax_rur = max(T_rur[days*24:(days+1)*24])
            TDmax_urb = max(T_urb[days*24:(days+1)*24])
            TmaxD.append(TDmax_urb-TDmax_rur)
            days += 1
            if days == nday:
                break
        m += 1
        sd = 1
    return st.mean(TmaxD[warmingdays:]), TmaxD

def MEAN_TOT(data, nday, warmingdays, typ):
    hours = 0 + warmingdays*24
    new_data = []

    for i in range(hours, nday*24):
        new_data.append(data[i])
    if typ == "mean":
        mean = np.mean(new_data)
        return mean
    elif typ == "sum":
        tot = np.sum(new_data)
        return tot
    else:
        print("typ ERROR")
        

#%%          
def create_runUWG_readEPW(i, X):
    results = os.path.join(project_folder, "results")  

    model_json = os.path.join(project_folder,"custom_uwg_param.json")
    
    with open(model_json) as json_file:
        data = json.load(json_file)
    
    new_data, k = data_input1(i, data, X)
    new_data = data_input2(new_data)
    new_data = data_input3(new_data)
    
    json_object = json.dumps(new_data, indent = 4)  
    with open('data.json', 'w') as outfile:
        json.dump(new_data, outfile)
    try:
        model = UWG.from_dict(new_data, epw)  
        model.generate()        
        model.epw_precision = 3
        with open('data1.json', 'w') as outfile:
            json.dump(model.to_dict(), outfile) 
        model.simulate()             
        #epw_output_name = str(i) + ".epw"        
        #epw_output = os.path.join(results, epw_output_name)
        
        # Write the simulation result to a file.
        CanTemp = []
        UBL = []
        RurTemp = []
        RoofTemp = []
        #CeilTemp = []
        WallTemp = []
        UCM_Qubl = []
        UCM_Qhvac = []
        UCM_Qtraffic = []
        UCM_Qroof = []
        UCM_Qwall = []
        UCM_coolConsump0 = [] #edificio n.1
        UCM_sensCoolDemand0 = [] #edificio n.1
        UCM_dehumDemand0 = [] #edificio n.1
        #ExtRoofFlux = []
        #IntRoofFlux = []
        #indoorTemp = []
    
        for j in range(len(model.UCMData)):
            #j sono le ore
            CanTemp.append(model.UCMData[j].canTemp - 273.15)
            UBL.append(model.UBLData[j].ublTemp - 273.15)
            RurTemp.append(model.WeatherData[j].temp - 273.15)
            RoofTemp.append(model.UCMData[j].roofTemp - 273.15) # Average (among building typologies) roof temperature
            #CeilTemp.append(model.UCMData[j].ceilTemp - 273.15) # Average (among building typologies) ceil temperature
            WallTemp.append(model.UCMData[j].wallTemp - 273.15) # Average wall temperature
            UCM_Qubl.append(model.UCMData[j].Q_ubl) # Convective heat exchange with UBL layer
            UCM_Qhvac.append(model.UCMData[j].Q_hvac) # Sensible heat flux from HVAC waste
            UCM_Qtraffic.append(model.UCMData[j].Q_traffic) # Sensible heat flux from HVAC waste
            UCM_Qroof.append(model.UCMData[j].Q_roof) # Sensible heat flux from building roof (convective)
            UCM_Qwall.append(model.UCMData[j].Q_wall) # Sensible heat flux from building wall (convective)
            #for i in range(len(model.UCMData[j].coolConsump)):
            #  UCM_coolConsump[i] = model.UCMData[j].coolConsump[i] # Cooling Energy Consumption in each building type [W/m2 per bld footprint]
            UCM_coolConsump0.append(model.UCMData[j].coolConsump0)
            UCM_sensCoolDemand0.append(model.UCMData[j].sensCoolDemand0)
            UCM_dehumDemand0.append(model.UCMData[j].dehumDemand0)
            #ExtRoofFlux.append(model.UCMData[j].ExtRoofFlux) # Average external roof surface heat flux (roof to outside) [W/m2 of roof area](parametro aggiunto)
            #IntRoofFlux.append(model.UCMData[j].IntRoofFlux) # Average internal roof surface heat flux (roof to inside) [W/m2 of roof area] (parametro aggiunto)
            #indoorTemp.append(model.UCMData[j].indoorTemp - 273.15) #Average (among building typologies) indoor temperature (aggiunto)
    
        Tdata = CanTemp
        
        T_profile = np.array(Tdata)
        Trur_profile = np.array(RurTemp)
        #Tindoorprofile = np.array(indoorTemp)
        Tdata_mean = np.mean(Tdata, axis = 0)
        Tdata_var = np.var(Tdata, axis = 0)
        Tdata_std = np.std(Tdata, axis = 0)
        Tdata_sum = np.sum(Tdata, axis = 0)
        Tdata_max = np.max(Tdata, axis = 0)
        Tdata_min = np.min(Tdata, axis = 0)
        
        Tdata_sorted = np.sort(Tdata, axis = 0)
        Tdata_sortmax5pc = Tdata_sorted[int(len(Tdata_sorted)*0.95):]
        Tdata_max5pc = np.mean(Tdata_sortmax5pc, axis = 0)
        Tdata_sortmin5pc = Tdata_sorted[:int(len(Tdata_sorted)*0.05)]
        Tdata_min5pc = np.mean(Tdata_sortmin5pc, axis = 0)
        
        uhii = UHII(RurTemp, Tdata, data['month'], data['day'], data['nday'], data["#warming_days"])
        tmaxd = TmaxD(RurTemp, Tdata, data['month'], data['day'], data['nday'], data["#warming_days"])  
        Qhvac = MEAN_TOT(UCM_Qhvac, data['nday'], data["#warming_days"], typ="mean")
        Qtraffic = MEAN_TOT(UCM_Qtraffic, data['nday'], data["#warming_days"], typ="mean")       
        #coolConsump0 = [0,0]
        #for i in range(len(model.UCMData[j].coolConsump)):              
        #    coolConsump[i] = MEAN(UCM_coolConsump[i], data['nday'], data["#warming_days"])
        coolConsump0 = MEAN_TOT(UCM_coolConsump0, data['nday'], data["#warming_days"], typ="sum")
        sensCoolDemand0 = MEAN_TOT(UCM_sensCoolDemand0, data['nday'], data["#warming_days"], typ="sum")
        dehumDemand0 = MEAN_TOT(UCM_dehumDemand0, data['nday'], data["#warming_days"], typ="sum")
                                   
        bldWidth = model.UCMData[0].bldWidth
        canWidth = model.UCMData[0].canWidth
        verToHor = model.UCMData[0].verToHor
                           
        results = [Tdata_mean, Tdata_var, Tdata_std, Tdata_sum, Tdata_max, Tdata_min, 
                   Tdata_max5pc, Tdata_min5pc, 
                   uhii[0], uhii[2], tmaxd[0], uhii[3], 
                   Qhvac, Qtraffic, coolConsump0, sensCoolDemand0, dehumDemand0,
                   bldWidth, canWidth, verToHor,
                   T_profile, uhii[1], tmaxd[1], #Tindoorprofile, 
                   uhii[4], Trur_profile
                   ]
        
        return results
    except:
        print("Simulation n."+str(i)+" ended with an ERROR") 

def create_runUWG_readEPW_test(i, X):
    results = os.path.join(project_folder, "results")  

    model_json = os.path.join(project_folder,"custom_uwg_param.json")
    
    with open(model_json) as json_file:
        data = json.load(json_file)
    
    new_data, k = data_input1(i, data, X)
    new_data = data_input2(new_data)
    new_data = data_input3(new_data)
    
    json_object = json.dumps(new_data, indent = 4)  
    #with open('data.json', 'w') as outfile:
    #    json.dump(new_data, outfile)
    #try:
    model = UWG.from_dict(new_data, epw)  
    model.generate()        
    model.epw_precision = 3
    with open('data1.json', 'w') as outfile:
        json.dump(model.to_dict(), outfile) 
    model.simulate()             
    #epw_output_name = str(i) + ".epw"        
    #epw_output = os.path.join(results, epw_output_name)
    
    # Write the simulation result to a file.
    CanTemp = []
    UBL = []
    RurTemp = []
    RoofTemp = []
    #CeilTemp = []
    WallTemp = []
    UCM_Qubl = []
    UCM_Qhvac = []
    UCM_Qtraffic = []
    UCM_Qroof = []
    UCM_Qwall = []
    UCM_coolConsump0 = [] #edificio n.1
    UCM_sensCoolDemand0 = [] #edificio n.1
    UCM_dehumDemand0 = [] #edificio n.1
    #ExtRoofFlux = []
    #IntRoofFlux = []
    #indoorTemp = []

    for j in range(len(model.UCMData)):
        #j sono le ore
        CanTemp.append(model.UCMData[j].canTemp - 273.15)
        UBL.append(model.UBLData[j].ublTemp - 273.15)
        RurTemp.append(model.WeatherData[j].temp - 273.15)
        RoofTemp.append(model.UCMData[j].roofTemp - 273.15) # Average (among building typologies) roof temperature
        #CeilTemp.append(model.UCMData[j].ceilTemp - 273.15) # Average (among building typologies) ceil temperature
        WallTemp.append(model.UCMData[j].wallTemp - 273.15) # Average wall temperature
        UCM_Qubl.append(model.UCMData[j].Q_ubl) # Convective heat exchange with UBL layer
        UCM_Qhvac.append(model.UCMData[j].Q_hvac) # Sensible heat flux from HVAC waste
        UCM_Qtraffic.append(model.UCMData[j].Q_traffic) # Sensible heat flux from HVAC waste
        UCM_Qroof.append(model.UCMData[j].Q_roof) # Sensible heat flux from building roof (convective)
        UCM_Qwall.append(model.UCMData[j].Q_wall) # Sensible heat flux from building wall (convective)
        #for i in range(len(model.UCMData[j].coolConsump)):
        #  UCM_coolConsump[i] = model.UCMData[j].coolConsump[i] # Cooling Energy Consumption in each building type [W/m2 per bld footprint]
        UCM_coolConsump0.append(model.UCMData[j].coolConsump0)
        UCM_sensCoolDemand0.append(model.UCMData[j].sensCoolDemand0)
        UCM_dehumDemand0.append(model.UCMData[j].dehumDemand0)
        #ExtRoofFlux.append(model.UCMData[j].ExtRoofFlux) # Average external roof surface heat flux (roof to outside) [W/m2 of roof area](parametro aggiunto)
        #IntRoofFlux.append(model.UCMData[j].IntRoofFlux) # Average internal roof surface heat flux (roof to inside) [W/m2 of roof area] (parametro aggiunto)
        #indoorTemp.append(model.UCMData[j].indoorTemp - 273.15) #Average (among building typologies) indoor temperature (aggiunto)

    Tdata = CanTemp
    
    T_profile = np.array(Tdata)
    Trur_profile = np.array(RurTemp)
    #Tindoorprofile = np.array(indoorTemp)
    Tdata_mean = np.mean(Tdata, axis = 0)
    Tdata_var = np.var(Tdata, axis = 0)
    Tdata_std = np.std(Tdata, axis = 0)
    Tdata_sum = np.sum(Tdata, axis = 0)
    Tdata_max = np.max(Tdata, axis = 0)
    Tdata_min = np.min(Tdata, axis = 0)
    
    Tdata_sorted = np.sort(Tdata, axis = 0)
    Tdata_sortmax5pc = Tdata_sorted[int(len(Tdata_sorted)*0.95):]
    Tdata_max5pc = np.mean(Tdata_sortmax5pc, axis = 0)
    Tdata_sortmin5pc = Tdata_sorted[:int(len(Tdata_sorted)*0.05)]
    Tdata_min5pc = np.mean(Tdata_sortmin5pc, axis = 0)
    
    uhii = UHII(RurTemp, Tdata, data['month'], data['day'], data['nday'], data["#warming_days"])
    tmaxd = TmaxD(RurTemp, Tdata, data['month'], data['day'], data['nday'], data["#warming_days"])  
    Qhvac = MEAN_TOT(UCM_Qhvac, data['nday'], data["#warming_days"], typ="mean")
    Qtraffic = MEAN_TOT(UCM_Qtraffic, data['nday'], data["#warming_days"], typ="mean")       
    #coolConsump0 = [0,0]
    #for i in range(len(model.UCMData[j].coolConsump)):              
    #    coolConsump[i] = MEAN(UCM_coolConsump[i], data['nday'], data["#warming_days"])
    coolConsump0 = MEAN_TOT(UCM_coolConsump0, data['nday'], data["#warming_days"], typ="sum")
    sensCoolDemand0 = MEAN_TOT(UCM_sensCoolDemand0, data['nday'], data["#warming_days"], typ="sum")
    dehumDemand0 = MEAN_TOT(UCM_dehumDemand0, data['nday'], data["#warming_days"], typ="sum")
                               
    bldWidth = model.UCMData[0].bldWidth
    canWidth = model.UCMData[0].canWidth
    verToHor = model.UCMData[0].verToHor
                       
    results = [Tdata_mean, Tdata_var, Tdata_std, Tdata_sum, Tdata_max, Tdata_min, 
               Tdata_max5pc, Tdata_min5pc, 
               uhii[0], uhii[2], tmaxd[0], uhii[3], 
               Qhvac, Qtraffic, coolConsump0, sensCoolDemand0, dehumDemand0,
               bldWidth, canWidth, verToHor,
               T_profile, uhii[1], tmaxd[1], #Tindoorprofile, 
               uhii[4], Trur_profile
               ]

    return results
    #except:
    #    print("Simulation n."+str(i)+" ended with an ERROR") 
                    
print("test")
create_runUWG_readEPW_test(8, X)
if input("Continue? [y/n] ") == "y":
        pass
else:
  sys.exit()
  
# %% 
if __name__ == '__main__':
    IO_matrix = pd.read_excel(os.path.join(results, "I-O_matrix.xlsx"), index_col=0, engine='openpyxl')
    # np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    start_time = time.time()  
    synthetic_results = parmap.map(create_runUWG_readEPW, range(len(X)), X, pm_pbar=True, pm_processes=num_CPUs)
    print("--- %s seconds for running" % round((time.time() - start_time), 2), (len(X)),"uwg simulations ---" )   
    # print("Plotting results...")
    
#%%

if __name__ == '__main__':    
    IO_matrix.insert(D, 'Tmean', 0) #adding output values in the I-O matrix
    IO_matrix.insert(D+1, 'Tvar', 0)
    IO_matrix.insert(D+2, 'Tstd', 0)
    IO_matrix.insert(D+3, 'Tsum', 0)
    IO_matrix.insert(D+4, 'Tmax', 0)
    IO_matrix.insert(D+5, 'Tmin', 0)
    IO_matrix.insert(D+6, 'Tmean5max', 0)
    IO_matrix.insert(D+7, 'Tmean5min', 0)
    IO_matrix.insert(D+8, 'UHII', 0)
    IO_matrix.insert(D+9, 'CDDD', 0)
    IO_matrix.insert(D+10, 'UHIImax', 0)
    IO_matrix.insert(D+11, 'UCII', 0)
    IO_matrix.insert(D+12, 'Qhvac', 0)
    IO_matrix.insert(D+13, 'Qtraffic', 0)
    IO_matrix.insert(D+14, 'coolConsump0', 0)
    IO_matrix.insert(D+15, 'sensCoolDemand0', 0)
    IO_matrix.insert(D+16, 'dehumDemand0', 0)
    IO_matrix.insert(D+17, 'bldWidth', 0)
    IO_matrix.insert(D+18, 'canWidth', 0)
    IO_matrix.insert(D+19, 'verToHor', 0)

    IO_matrix['Tmean'] = np.array(synthetic_results, dtype="object")[:,0]
    IO_matrix['Tvar'] = np.array(synthetic_results, dtype="object")[:,1]  
    IO_matrix['Tstd'] = np.array(synthetic_results, dtype="object")[:,2]  
    IO_matrix['Tsum'] = np.array(synthetic_results, dtype="object")[:,3] 
    IO_matrix['Tmax'] = np.array(synthetic_results, dtype="object")[:,4]
    IO_matrix['Tmin'] = np.array(synthetic_results, dtype="object")[:,5]
    IO_matrix['Tmean5max'] = np.array(synthetic_results, dtype="object")[:,6]
    IO_matrix['Tmean5min'] = np.array(synthetic_results, dtype="object")[:,7]
    IO_matrix['UHII'] = np.array(synthetic_results, dtype="object")[:,8]
    IO_matrix['CDDD'] = np.array(synthetic_results, dtype="object")[:,9]
    IO_matrix['UHIImax'] = np.array(synthetic_results, dtype="object")[:,10]
    IO_matrix['UCII'] = np.array(synthetic_results, dtype="object")[:,11]
    IO_matrix['Qhvac'] = np.array(synthetic_results, dtype="object")[:,12]
    IO_matrix['Qtraffic'] = np.array(synthetic_results, dtype="object")[:,13]
    IO_matrix['coolConsump0'] = np.array(synthetic_results, dtype="object")[:,14]
    IO_matrix['sensCoolDemand0'] = np.array(synthetic_results, dtype="object")[:,15]
    IO_matrix['dehumDemand0'] = np.array(synthetic_results, dtype="object")[:,16]
    IO_matrix['bldWidth'] = np.array(synthetic_results, dtype="object")[:,17]
    IO_matrix['canWidth'] = np.array(synthetic_results, dtype="object")[:,18]
    IO_matrix['verToHor'] = np.array(synthetic_results, dtype="object")[:,19]
    
    IO_matrix.to_excel(os.path.join(results, "I-O_matrix.xlsx"), 
                          index = True)
    
    #%%
import pandas as pd
import os
project_folder = os.path.dirname(__file__) 
results = os.path.join(project_folder, "results")
IO_matrix = pd.read_excel(os.path.join(results, "I-O_matrix.xlsx"), index_col=0, engine='openpyxl')

#%%
# if __name__ == '__main__':  
    # import seaborn as sns
    # #sns.pairplot(IO_matrix, vars=['UHII','CDDD'])
    # plt.style.use('default')
    # for name in variables_names:
        # g = sns.jointplot(data=IO_matrix, x=name, y='UHII', kind="reg")
        # g.plot_joint(sns.scatterplot, s=100, alpha=.5)
        # g.savefig(os.path.join(results, 'UHII-'+str(name)+'_N='+str(N)+'.png'))
        # plt.close()
        # g = sns.jointplot(data=IO_matrix, x=name, y='UHIImax', kind="reg")
        # g.plot_joint(sns.scatterplot, s=100, alpha=.5)
        # g.savefig(os.path.join(results, 'UHIImax-'+str(name)+'_N='+str(N)+'.png'))
        # plt.close()

#%%
#
########### Sensitivity #######################################
#
import numpy as np

def Sobol_Sensitivity(ouput_values, name):
    # Perform analysis
    Si = sobol.analyze(problem, np.array(ouput_values), 
                       calc_second_order=data["#calc_second_order"], print_to_console=False) #check
    
#    plt.style.use('default')
#    plt.figure()
#    fig, ax = plt.subplots()
#    y = np.arange(D)
#    ax.barh(y, Si['S1'], align='center', xerr=Si['S1_conf'])
#    ax.set_yticks(y)
#    ax.set_yticklabels(variables_names)
#    ax.invert_yaxis()
#    ax.set_xlabel('First order Sensitivity Indeces for '+str(name))
#    plt.savefig(os.path.join(results, 'S1_plot_N='+str(N)+'_'+str(name)+'.png'), bbox_inches='tight')
#    plt.close()
    
    if data["#calc_second_order"] == True: 
        Si_2 = {'S1': Si['S1'].tolist(),
                  'S1_conf': Si['S1_conf'].tolist(),
                  'ST': Si['ST'].tolist(),
                  'ST_conf': Si['ST_conf'].tolist(),
                  # 'S2': Si['S2'].tolist(),
                  # 'S2_conf': Si['S2_conf'].tolist()
                  }
    else:
        Si_2 = {'S1': Si['S1'].tolist(),
                  'S1_conf': Si['S1_conf'].tolist()
                  }       
    
    Si_pd = pd.DataFrame.from_dict(Si_2)
    
    Si_pd.to_excel(os.path.join(results, 'SensitivityIndexes'+str(N)+str(name)+'.xlsx'), index = True)
    
def Morris_Sensitivity(ouput_values, name):
    # Perform analysis
    num_levels = 4
    output = np.array(ouput_values)
    Si = M.analyze(problem, X, output, conf_level=0.95, print_to_console=False, num_levels=num_levels) 
    num_vars = problem['num_vars']
    
    #computing additional median_star and median_star_conf values
    delta = _compute_delta(num_levels)
    num_trajectories = int(output.size / (num_vars + 1))
    ee = np.zeros((num_vars, num_trajectories))
    ee = compute_elementary_effects(
            X, output, int(output.size / num_trajectories), delta) 

    Si['mu_test'] = np.average(ee, 1)
    Si['median_star'] = np.median(np.abs(ee), 1)        
    Si['median_star_conf'] = [None] * num_vars #https://www.sixsigmain.it/ebook/Capu4-8.html#:~:text=L'intervallo%20di%20confidenza%20della%20mediana%20calcolato%20con%20il%20metodo,intervallo%20%C3%A8%20maggiore%20del%2025%25.&text=Il%20minimo%20e%20il%20massimo,intervallo%20di%20confidenza%20della%20mediana.
    for j in range(num_vars):
        Si['median_star_conf'][j] = Si['mu_star_conf'][j]*1.25    
    
    Si_pd = pd.DataFrame.from_dict(Si)
    
    Si_pd.to_excel(os.path.join(results, 'Morris_results'+str(N)+str(name)+'.xlsx'), index = True)

if __name__ == '__main__':
    if data["#sensitivity"] == "Sobol":
        Sobol_Sensitivity(ouput_values=IO_matrix['Tmean'], name='Tmean')
        Sobol_Sensitivity(ouput_values=IO_matrix['Tmax'], name='Tmax')
        Sobol_Sensitivity(ouput_values=IO_matrix['Tmin'], name='Tmin')
        Sobol_Sensitivity(ouput_values=IO_matrix['Tmean5max'], name='Tmean5max')
        Sobol_Sensitivity(ouput_values=IO_matrix['Tmean5min'], name='Tmean5min')
        Sobol_Sensitivity(ouput_values=IO_matrix['UHII'], name='UHII')
        Sobol_Sensitivity(ouput_values=IO_matrix['CDDD'], name='CDDD')
        Sobol_Sensitivity(ouput_values=IO_matrix['UHIImax'], name='UHIImax')
        Sobol_Sensitivity(ouput_values=IO_matrix['Qhvac'], name='Qhvac')
        Sobol_Sensitivity(ouput_values=IO_matrix['coolConsump0'], name='coolConsump0')
        Sobol_Sensitivity(ouput_values=IO_matrix['sensCoolDemand0'], name='sensCoolDemand0')
        Sobol_Sensitivity(ouput_values=IO_matrix['dehumDemand0'], name='dehumDemand0')
    if data["#sensitivity"] == "Morris":
        Morris_Sensitivity(ouput_values=IO_matrix['Tmean'], name='Tmean')
        Morris_Sensitivity(ouput_values=IO_matrix['Tmax'], name='Tmax')
        Morris_Sensitivity(ouput_values=IO_matrix['Tmin'], name='Tmin')
        Morris_Sensitivity(ouput_values=IO_matrix['Tmean5max'], name='Tmean5max')
        Morris_Sensitivity(ouput_values=IO_matrix['Tmean5min'], name='Tmean5min')
        Morris_Sensitivity(ouput_values=IO_matrix['UHII'], name='UHII')
        Morris_Sensitivity(ouput_values=IO_matrix['CDDD'], name='CDDD')
        Morris_Sensitivity(ouput_values=IO_matrix['UHIImax'], name='UHIImax')
        Morris_Sensitivity(ouput_values=IO_matrix['Qhvac'], name='Qhvac')
        Morris_Sensitivity(ouput_values=IO_matrix['coolConsump0'], name='coolConsump0')
        Morris_Sensitivity(ouput_values=IO_matrix['sensCoolDemand0'], name='sensCoolDemand0')
        Morris_Sensitivity(ouput_values=IO_matrix['dehumDemand0'], name='dehumDemand0')
        
#%%
if __name__ == '__main__':
    if N < 32 :   
        T_profiles = pd.DataFrame() 
        plot = pd.DataFrame()
        for i in range(len(X)):
            panda = pd.DataFrame(np.array(synthetic_results, dtype="object")[:,20][i])
            T_profiles = T_profiles.append(panda.T, ignore_index=True)       
            #plot = pd.concat([plot, panda], ignore_index=False, axis=1)
        T_profiles.to_excel(os.path.join(results, "T_profiles.xlsx"), 
                              index = True)  
         
        UHII_profiles = pd.DataFrame()
        for i in range(len(X)):
            panda = pd.DataFrame(np.array(synthetic_results, dtype="object")[:,21][i])
            UHII_profiles = UHII_profiles.append(panda.T, ignore_index=True)
        UHII_profiles.to_excel(os.path.join(results, "UHII_profiles.xlsx"), 
                              index = True)  
        
        TmaxD_profiles = pd.DataFrame()
        #print(pd.DataFrame(np.array(synthetic_results)[:,13][0]))
        for i in range(len(X)):
            panda = pd.DataFrame(np.array(synthetic_results, dtype="object")[:,22][i])
            TmaxD_profiles = TmaxD_profiles.append(panda.T, ignore_index=True)
        #print(TmaxD_profiles)
        TmaxD_profiles.to_excel(os.path.join(results, "TmaxD_profiles.xlsx"), 
                              index = True)
        
        #Tindoorprofiles = pd.DataFrame() 
        #plot = pd.DataFrame()
        #for i in range(len(X)):
        #    panda = pd.DataFrame(np.array(synthetic_results, dtype="object")[:,15][i])
            #Tindoorprofiles = Tindoorprofiles.append(panda.T, ignore_index=True)       
            #plot = pd.concat([plot, panda], ignore_index=False, axis=1)
        #Tindoorprofiles.to_excel(os.path.join(results, "Tindoor_profiles.xlsx"), index = True)  
        
        UCII_profiles = pd.DataFrame() 
        plot = pd.DataFrame()
        for i in range(len(X)):
            panda = pd.DataFrame(np.array(synthetic_results, dtype="object")[:,23][i])
            UCII_profiles = UCII_profiles.append(panda.T, ignore_index=True)       
            #plot = pd.concat([plot, panda], ignore_index=False, axis=1)
        UCII_profiles.to_excel(os.path.join(results, "UCII_profiles.xlsx"), 
                              index = True) 
        
        Trur_profile = pd.DataFrame() 
        plot = pd.DataFrame()
        for i in range(len(X)):
            panda = pd.DataFrame(np.array(synthetic_results, dtype="object")[:,24][i])
            Trur_profile = Trur_profile.append(panda.T, ignore_index=True)       
            #plot = pd.concat([plot, panda], ignore_index=False, axis=1)
        Trur_profile.to_excel(os.path.join(results, "Trur_profile.xlsx"), 
                              index = True) 
    print("...End")
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 17:47:23 2020

@author: Sam
"""
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.stats 
from sklearn.preprocessing import MinMaxScaler

#build dataset of authoritarian govt indicators ~Eventually unused, opted to use democracy index from the Economist
pd_auth = pd.read_csv("wvs_and_more.csv")
auth_indicators = pd_auth.loc[:,['V60', 'V61', 'V65', 'V79', 'V95', 'V17', 'V128', 'V129'
                                 , 'V130', 'V133', 'V135', 'V136', 'V137', 'V140', 'V141',
                                 'V142', 'V173', 'V186', 'V228A', 'V228B',
 'V228C', 'V228E', 'V228F', 'V228G', 'V228H', 'V228I']]
countries_series = pd_auth.loc[:,'country_name']
auth_indicators.insert(0,'country',countries_series)
auth_indicators.to_csv('wvs_auth_indicators.csv')


#clean and create relevant Economic Indicators
#econ_pd = pd.read_csv("WEOEconomic_indicators.csv")

indicators_df = econ_pd.loc[:,['Country','WEO Subject Code','2018','2019','2020']]
#remain with relevant row entries
indicators_df = indicators_df.loc[indicators_df['WEO Subject Code'].isin(
        ['NGDP_RPCH','NGDPRPPPPCPCH','PCPIPCH','LUR','GGXCNL_NGDP'])]
indicators_df.set_index('WEO Subject Code', inplace = True)
#rename rows to readable entries
indicators_df.rename(index={'NGDP_RPCH':'GDP', 'NGDPRPPPPCPCH': 'GDP_per_capita','PCPIPCH':'inflation',
                            'LUR':'unemployment_rate','GGXCNL_NGDP':'net_lending'}, inplace = True)



##obtain mobility data
#read in csv as mobility_df
mobility_df = pd.read_csv(mobility_path) #"global_mobility_report.csv
#obtain percentage change 30 days from baseline
  mobility_df['date'] = pd.to_datetime(mobility_df['date'])
  start_date = '2020-02-15'
  start_date = datetime.strptime(start_date, "%Y-%m-%d")
  end_date = '2020-03-16'
  end_date = datetime.strptime(end_date, "%Y-%m-%d")
  date_range = (mobility_df['date']>=start_date) & (mobility_df['date']<=end_date)
  mobility_df = mobility_df.loc[date_range]
  
##Obtain only countrywide data
mobility_df = mobility_df.loc[(mobility_df.sub_region_1.isnull()) & 
                (mobility_df.sub_region_2.isnull())]  

#housekeeping -> rename columns + average out mobility rates across countries 
 mobility_df.drop(columns = ['country_region_code', 'sub_region_1', 'sub_region_2'], inplace = True)
mobility_df.rename(columns={'country_region':'country'}, inplace = True)
#obtain global averages for first 30 days
global_mobility_averages = mobility_df.groupby('country').mean()
global_mobility_averages.to_csv("global_mobility_averages.csv") 
 
#Join democracy + mobility datasets
 average_mobility_df = pd.read_csv("global_mobility_averages.csv")
 pd.read_csv("democracy_index_2019.csv")
 joined_df = pd.merge(average_mobility_df,democracy_df, on='country',how='inner')
 
#Correlations & p-value
 correlations_matrix = joined_df.corr(method = 'pearson')
 correlations_matrix.to_csv("dem_mob_corr.csv")
 scipy.stats.ttest_ind(joined_df.workplaces_percent_change_from_baseline, 
                       joined_df.democracy_score,equal_var=False)
 
 
##General mobility trends across countries measured over time (duration = 3 months)
all_mobility_df = pd.read_csv(mobility_path) #global_mobility_report.csv
 
all_mobility_df = all_mobility_df.loc[(all_mobility_df.sub_region_1.isnull()) & 
                (all_mobility_df.sub_region_2.isnull())]  
 
all_mobility_df['date'] =  pd.to_datetime(all_mobility_df['date'])
  start_date = '2020-02-15'
 start_date = datetime.strptime(start_date, "%Y-%m-%d")
  end_date = '2020-05-16'
 end_date = datetime.strptime(end_date, "%Y-%m-%d")
 date_range = (all_mobility_df['date']>=start_date) & (all_mobility_df['date']<=end_date)
 all_mobility_df = all_mobility_df.loc[date_range]
 
all_mobility_df.drop(columns = ['country_region_code', 'sub_region_1', 'sub_region_2'], inplace = True)
all_mobility_df.rename(columns={'country_region':'country'}, inplace = True)
 
#get averages
all_mobility_averages = all_mobility_df.groupby('country').mean()
all_mobility_averages.to_csv("all_mobility_averages.csv")


#read in ILO data
ilo_df = pd.read_csv(ilo_path)
#get national data
national_data = (ilo_df['sex'] == 'Total') & (ilo_df['area_type'] == 'National')
ilo_df = ilo_df.loc[national_data]
#get only 2020 data
ilo_df = ilo_df.loc[ilo_df['year'] == 2020]
#housekeeping -> drop + rename columns
ilo_df.drop(columns=['ref_area','agriculture','industry','services','sex'], inplace = True)
ilo_df.reset_index(inplace=True)
ilo_df.rename(columns={'ref_area.label':'country'}, inplace=True)
ilo_df.to_csv("ilostats_ref.csv")
 
##Get ILO, work from home data
country_work_home = pd.read_csv(work_home_path) #use country_workathome.csv
internet_df = pd.read_csv(internet_access_path) #use all_mobility_averages.csv
ilo_ref_df = pd.read_csv(ilo_ref_path)  #use ilostats_ref.csv

#housekeeping -> drop + rename columns 
country_work_home.drop(columns=['country_code','year_ilo'], inplace=True)
internet_df.drop(columns=['2017', 'Unnamed: 3'],inplace=True)
internet_df.rename(columns={'2018':'internet_penetration'},inplace=True)
 
#merge obtained datasets: ILO data, work from home data and internet penetration datasets
 
core_indicators_df = pd.merge(ilo_ref_df,all_mobility_df,on='country',how='left')
core_indicators_df = pd.merge(core_indicators_df,internet_df, on='country',how='left')
core_indicators_df = pd.merge(core_indicators_df,country_work_home,on='country',how='left')
 
#Housekeeping
core_indicators_df.drop(columns=['Unnamed: 0'],inplace=True) 
core_indicators_df.set_index('country',inplace=True)
core_indicators_df.to_csv("macro_mobility.csv")

''' 
Analysis
'''

#Check income groups for changes in mobility
work_df.groupby('wb_income_group.label').workplaces_percent_change_from_baseline.mean()

#Check income groups for industry composition & internet penetration
work_df.groupby('wb_income_group.label').agriculture_percent.mean()
work_df.groupby('wb_income_group.label').industry_percent.mean()
work_df.groupby('wb_income_group.label').services_percent.mean()
work_df.groupby('wb_income_group.label').teleworkable.mean()
work_df.groupby('wb_income_group.label').internet_penetration.mean()


#merge health data 
health_df = pd.read_csv(health_path)   #read health data from covid_health_data.csv (owid)
#get relevant data
health_df = health_df.loc[:,['location','date','total_cases','new_cases','total_deaths','stringency_index','cvd_death_rate']]

#get week with max growth rate of cases for each country
health_df_copy = health_df.copy()
health_df_copy['date'] = pd.to_datetime(health_df_copy['date'])
health_df_copy.date = health_df_copy['date'] - pd.to_timedelta(7, unit='d')
health_df_copy = health_df_copy.groupby(['location',pd.Grouper(key='date',freq='W-MON')])['new_cases'].sum()
max_growth_rates = health_df_copy.max(level='location')

#get total deaths and total cases in each country
covid_deaths = health_df.groupby(by='location')['total_deaths'].max()
covid_deaths['death_rate'] = covid_deaths.total_deaths/covid_cases.total_cases
covid_cases = health_df.groupby('location')['total_cases'].max()

#average stringency index, non-zero and non-null entries
stringencies = health_df.groupby(by='location').apply(lambda df: df.loc[(pd.notna(df.stringency_index)) & (df.stringency_index!=0)].mean())
stringencies_copy = stringencies.copy()
stringencies_copy = stringencies_copy.loc[stringencies_copy.level_1.isin(['stringency_index'])]
stringencies_copy.rename(columns={0:'avg_stringency_index'},inplace=True)
stringencies_copy.drop(columns=['level_1'],inplace=True)
stringencies = stringencies_copy.reset_index()
stringencies.drop(columns=['index'],inplace=True)
stringencies.rename(columns={'location':'country'},inplace=True)

#dataframes and housekeeping
max_growth_rates = max_growth_rates.reset_index()
max_growth_rates.rename(columns={'location':'country'},inplace=True)
covid_deaths = covid_deaths.reset_index()
covid_deaths.rename(columns={'location':'country'},inplace=True)
covid_cases.reset_index()
covid_cases.rename(columns={'location':'country'},inplace=True)

#get recent gdp data & clean
gdp_df = pd.read_csv("world_bank_gdp.csv")
gdp_df = gdp_df.loc[:,['Country Name','2018']]
gdp_df.rename(columns={'Country Name': 'country'},inplace=True)
gdp_df.rename(columns={'2018':'gdppc_2018'},inplace=True)


#merge datasets: covid cases, growth rates, stringency levels and gdp per capita
macro_mobility_df = pd.read_csv(macro_mobility_path) #read in macro_mobility.csv
macro_mobility_health = pd.merge(macro_mobility_df,covid_cases,on='country',how='left')
macro_mobility_health = pd.merge(macro_mobility_health,covid_deaths,on='country',how='left')
macro_mobility_health = pd.merge(macro_mobility_health,max_growth_rates,on='country',how='left')
macro_mobility_health = pd.merge(macro_mobility_health,stringencies,on='country',how='left')
macro_mobility_health = pd.merge(macro_mobility_health,gdp_df,on='country',how='left')

##https://www.ilo.org/ilostat-files/Documents/description_ECO_EN.pdf for category splitting

#teleworkable workplace data
work_type_df = pd.read_csv("NAICS_workfromhome.csv")  #NAICS_workfromhome data + manual sector category via criteria - 
                                                    ##https://www.ilo.org/ilostat-files/Documents/description_ECO_EN.pdf
work_type_values = work_type_df.groupby(by='sector').mean()
#get average manual and labelled means for each sector
work_type_values['teleworkable_emp_avg'] = (work_type_values['teleworkable_manual_emp'] + work_type_values['teleworkable_emp'])/2
macro_mobility_health['teleworkable_ag']=macro_mobility_health['agriculture_percent']*work_type_values.teleworkable_emp_avg[0]
macro_mobility_health['teleworkable_ind']=macro_mobility_health['industry_percent']*work_type_values.teleworkable_emp_avg[1]
macro_mobility_health['teleworkable_services']=macro_mobility_health['services_percent']*work_type_values.teleworkable_emp_avg[2]
macro_mobility_health['all_teleworkable'] = macro_mobility_health.teleworkable_ag + macro_mobility_health.teleworkable_ind + macro_mobility_health.teleworkable_services


##creating resiliency index with stata; macro_mobility_health exported at end as macro_resiliency
#This part of the code was processed in stata to obtain principal components via PCA
'''
-> import macro_mobility_health.csv
#standardize data
. egen z_agriculture = std(agriculture_percent)

. egen z_industry = std(industry_percent)

. egen z_services = std(services_percent)

. egen z_workplaces = std(workplaces_percent_change_from_b)
(65 missing values generated)

. egen z_residential = std( residential_percent_change_from_)
(66 missing values generated)

. egen z_internet = std(internet_penetration)
(19 missing values generated)

. egen z_all_teleworkable = std(all_teleworkable)

. egen z_cases = std(total_cases)
(19 missing values generated)

. egen z_death_rate = std(death_rate)
(20 missing values generated)

. egen z_stringency = std(avg_stringency_index)
(37 missing values generated)

. egen z_gdp = std(gdppc_2018)

 pca z_agriculture z_industry z_services z_workplaces z_residential z_interne
> t z_all_teleworkable z_cases z_death_rate z_stringency z_gdp

rotate  #rotate components 
predict resiliency  #->predict resiliency #resiliency field has indexed data

export delimited using "macro_resiliency", replace

'''


##normalize resiliency values & include resiliency index in df
macro_resiliency = pd.read_csv("macro_resiliency.csv")
res_min = macro_resiliency.resiliency.min()
res_max = macro_resiliency.resiliency.max()
oldRange = res_max - res_min
newMax = 1 
newMin = 0
newRange = newMax - newMin
newVals = (((macro_resiliency.resiliency - res_min) * newRange)/oldRange) + newMin
macro_resiliency['resiliency_index'] = newVals*100

macro_resiliency.to_csv("all_macro_resiliency_data.csv")

##prepare relevant dataframes for correlations
##Relevant df's -> Mobility and by dominant sector (agriculture, industry services)

mobility_corrs = macro_mobility_health.loc[:,['residential_percent_change_from_baseline','workplaces_percent_change_from_baseline', 'retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline',     'parks_percent_change_from_baseline',  'transit_stations_percent_change_from_baseline']]
residential_corrs = macro_mobility_health.loc[:,['residential_percent_change_from_baseline','workplaces_percent_change_from_baseline','agriculture_percent','services_percent','industry_percent','max_growth_percent','avg_stringency_index','total_cases','total_cases_world_percent','death_rate','all_teleworkable','gdppc_2018']]
ag_df = macro_mobility_health.loc[macro_mobility_health.dominant_sector == 'agriculture_percent']
ag_df = ag_df.loc[:,['residential_percent_change_from_baseline','workplaces_percent_change_from_baseline','max_growth_percent','avg_stringency_index','total_cases','total_cases_world_percent','death_rate', 'all_teleworkable', 'internet_penetration','gdppc_2018']]

ind_df = macro_mobility_health.loc[macro_mobility_health.dominant_sector == 'industry_percent']
ind_df = ind_df.loc[:,['residential_percent_change_from_baseline','workplaces_percent_change_from_baseline','max_growth_percent','avg_stringency_index','total_cases','total_cases_world_percent','death_rate','all_teleworkable', 'internet_penetration','gdppc_2018']]

services_df = macro_mobility_health.loc[macro_mobility_health.dominant_sector == 'services_percent']
services_df = services_df.loc[:,['residential_percent_change_from_baseline','workplaces_percent_change_from_baseline','max_growth_percent','avg_stringency_index','total_cases','total_cases_world_percent','death_rate','all_teleworkable', 'internet_penetration','gdppc_2018']]

#get all correlations across regions by major employment industry 
mobility_corrs = mobility_corrs.corr()
relevant_correlations = residential_corrs.corr()
ag_corrs = ag_df.corr()
services_corrs = services_df.corr() #only 2 entries in industry df, 1 with NaN values so didn't check for correlations

##send correlations out to csv's
mobility_corrs.to_csv("mobility_corrs.csv")
relevant_correlations.to_csv("relevant_corrs.csv")
ag_corrs.to_csv("ag_correlations.csv")
services_corrs.to_csv("serv_corrs.csv")


#label Our World in Data covid data with region + income group
labels_df = macro_mobility_health.loc[:,['country', 'ilo_region.label','ilo_subregion_broad.label', 'ilo_subregion_detailed.label','wb_income_group.label']]
owid_df = pd.read_csv('owid-covid-data.csv')
owid_df.rename(columns={'location':'country'},inplace=True)
owid_labels = pd.merge(owid_df,labels_df,on='country',how='outer')
owid_labels.to_csv('owid_covid_data_labels.csv')

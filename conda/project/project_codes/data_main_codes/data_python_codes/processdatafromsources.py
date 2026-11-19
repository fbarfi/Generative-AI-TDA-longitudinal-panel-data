"""
International Relations Data Processing Module
===========================================

This module processes international relations data from multiple sources to generate 
a comprehensive dyadic (country-pair) dataset for international relations analysis.

Main Functionality
----------------
1. Loads and processes data from various international relations datasets (1885-2010)
2. Generates dyadic relationships and computes key variables
3. Handles data cleaning, transformation, and merging
4. Produces standardized output for analysis

Generated Variables
-----------------
Military and Security:
    - disp: Military disputes between countries (binary)
    - disp_lag: Lagged military disputes (binary)
    - majpower: Major power status (binary)
    - hegdef: Military expenditure ratio (continuous)
    - contgkf: Geographic contiguity (binary)

Economic and Trade:
    - smldep: Trade dependency (continuous)
    - lcaprat: Log capability ratio (continuous)

Political and Diplomatic:
    - allies: Alliance relationships (binary)
    - smldmat: Democratic/Autocratic regime characteristics (continuous)

Geographic:
    - lgmidist: Log distance between capitals (continuous)

Required Data Files
-----------------
Place in specified folder:
    1. system2016.csv: System of states data
    2. dyadic_format_v3.dta: Dyadic relationships
    3. Dyadic MID Dataset_short.csv: Military disputes
    4. dyadic_trade_COW_4.0.csv: Trade flows
    5. p5v2018.csv: Polity scores
    6. alliance_v4.1_by_dyad_yearly.csv: Alliance data
    7. NMC_5_0.csv: National material capabilities
    8. contdird.csv: Contiguity data
    9. capdist.csv: Capital distances
    10. COW country codes.csv: Country codes

Class Usage
----------
args = {
    'beginYear': 1946,    # Start year for analysis
    'endYear': 2010,      # End year for analysis
    'folder': 'data/'     # Path to data files
}

processor = ProcessDataFromSources(args)
data_sources = processor.reading_all_input_data()
final_data = processor.gen_all_vars(data_sources)

Key Methods
----------
reading_all_input_data():
    Loads and prepares all source datasets

gen_all_vars(data):
    Generates all variables and combines into final dataset

Variable Generation Methods:
    - gen_dyad(): Creates dyadic relationships
    - gen_majp_values(): Calculates major power status
    - gen_smldep(): Calculates trade dependency
    - gen_depend(): Processes trade relationships
    - gen_polity(): Processes regime characteristics
    - gen_allies(): Processes alliance relationships
    - gen_lcaprat(): Calculates capability ratios
    - gen_contgkf(): Processes contiguity data
    - gen_hegdef(): Calculates military expenditure ratios
    - gen_lgmidist(): Calculates distance measures
    - gen_disp(): Processes military disputes

Output Format
-----------
Returns pandas DataFrame with columns:
    - year: Year of observation
    - dyadid: Unique identifier for country pair (ccode1*1000 + ccode2)
    - ccode1, ccode2: Country codes (COW system)
    - Generated variables as listed above

Notes
-----
1. Uses Correlates of War (COW) country coding system
2. Handles both directed and undirected dyadic relationships
3. Includes data cleaning and missing value handling
4. Maintains temporal consistency across variables
5. Processes only valid country pairs within system membership

Dependencies
-----------
- numpy: Numerical computations
- pandas: Data manipulation and analysis
- scipy.stats: Statistical computations

Sources
-------
- Correlates of War Project (COW)
- Polity Project
- National Material Capabilities dataset

Author: Badredine Arfi

Last Updated: October 24, 2024
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


import numpy as np
import pandas as pd

class ProcessDataFromSources():

    def __init__(self,args): 
        
        super(ProcessDataFromSources,self).__init__()   
        self.args = args
        self.beginYear = args['beginYear']
        self.endYear = args['endYear']
        self.folder = args['folder']

    def gen_dyad(self, data_dyad):
        beginYear = self.beginYear
        endYear = self.endYear
        folder = self.folder
        
        data_dyad_1 = data_dyad.loc[(data_dyad['year'] >= beginYear ) 
                                    & (data_dyad['year'] <= endYear),
                                    :]
    
        df_dyad = pd.DataFrame({'dyadid' : 1000*data_dyad_1.ccode1.astype(int)+ 
                                data_dyad_1.ccode2.astype(int)})
        data_dyad_2 = pd.concat([df_dyad,data_dyad_1],axis=1)
        df_dyad = data_dyad_2[['year','dyadid','ccode1','ccode2']].sort_values(by \
                                                = ['year','dyadid']).reset_index(drop=True)
    
        return df_dyad.astype(int)

    """ 
    Generating "majpower" variable from year = 1885 to year = 2010
    """

    def gen_majp_values(self, ccode1,ccode2,year):
        
        majpower = 0
        if (ccode1 == 2)  | (ccode2 == 2) & (year > 1898):
            majpower =1
        elif (ccode1 == 200) | (ccode2 == 200):
            majpower = 1
        elif (ccode1 == 220) | (ccode2 == 220) & (year < 1941):
            majpower = 1
        elif (ccode1 == 220) | (ccode2 == 220) & (year > 1946):
            majpower = 1
        elif (ccode1 == 255) | (ccode2 == 255) & (year < 1919):
            majpower = 1
        elif (ccode1 == 255) | (ccode2 == 255) & (1924 < year):
            majpower = 1
        elif (ccode1 == 255) | (ccode2 == 255) & (year < 1946):
            majpower = 1
        elif (ccode1 == 255) | (ccode2 == 255) & (year > 1990):
            majpower = 1
        elif (ccode1 == 300) | (ccode2 == 300) & (year < 1919):
            majpower = 1
        elif (ccode1 == 325) | (ccode2 == 325) & (year > 1859):
            majpower = 1
        elif (ccode1 == 325) | (ccode2 == 325) & (year < 1944):
            majpower = 1
        elif (ccode1 == 365) | (ccode2 == 365) & (year < 1918):
            majpower = 1
        elif (ccode1 == 365) | (ccode2 == 365) & (year > 1922):
            majpower = 1
        elif (ccode1 == 710) | (ccode2 == 710) & (year > 1949):
            majpower = 1
        elif (ccode1 == 740) | (ccode2 == 740) & (1894 < year):
            majpower = 1
        elif (ccode1 == 740) | (ccode2 == 740) & (year < 1946):
            majpower = 1
        elif (ccode1 == 740) | (ccode2 == 740) & (year > 1990):
            majpower = 1
    
        return majpower


    def gen_majpower(self, data_dyad):
        
        n_years = pd.DataFrame({'year': pd.unique(data_dyad['year'])}).sort_values(by = ['year'])
        
        data_list = []
        for iyear in range(len(n_years)):
            data_dyad_year = data_dyad.loc[data_dyad.year \
                                        == n_years.year[iyear],:].reset_index(drop=True)
            
            maj_list, dyad_list, year_list, ccode1_list, ccode2_list = [], [], [] , [], []
            for i in range(len(data_dyad_year)):
                ccode_1 = data_dyad_year.ccode1[i]
                ccode_2 = data_dyad_year.ccode2[i]
                year_list.append(n_years.year[iyear])
                dyad_list.append(data_dyad_year.dyadid[i])
                ccode1_list.append(data_dyad_year.ccode1[i])
                ccode2_list.append(data_dyad_year.ccode2[i])
                maj_list.append(self.gen_majp_values(ccode_1,ccode_2,iyear))
                
            data_list.append(pd.DataFrame({'year' : year_list, 'dyadid': dyad_list, \
                            'ccode1':ccode1_list, 'ccode2':ccode2_list,'majpower': maj_list}))
            
        df_majpower = (pd.concat(data_list,axis=0)).reset_index(drop=True)
    
        return df_majpower
        

    """
    Generating variable "depend" from year = 1885 to year = 2010
    """

    def gen_smldep(self, flow1,flow2):
        return (flow1+flow2)/max(abs(flow1),abs(flow2))


    def gen_depend(self, data_dyad,data_trade):
        k_list = []
        p_years = pd.unique(data_trade['year'])
        for iyear in range(len(p_years)):
            data_trade_year = data_trade.loc[(data_trade.year == p_years[iyear]),:].reset_index(drop=True)
            data_dyad_year = data_dyad.loc[(data_dyad.year == p_years[iyear]),:].reset_index(drop=True)
            df_common_dyads = pd.DataFrame({'dyadid': 
                                            np.intersect1d(data_dyad_year.dyadid,data_trade_year.dyadid)})
           
            m_list = []
            dyad_list = []
            year_list = []
   
            for i in range(len(df_common_dyads)):
                m_list.append(self.gen_smldep(data_trade_year.flow1[i],data_trade_year.flow2[i]))
                year_list.append(p_years[iyear])
                dyad_list.append(df_common_dyads.dyadid[i])
   
            df_trade_year = pd.DataFrame({'year' : year_list, 'dyadid': dyad_list, 'smldep': m_list})
            df_trade_merge_0 = pd.merge(left = data_dyad_year,right = df_trade_year, \
                                        how ='outer', on = ['dyadid'] )
            df_trade_merge = df_trade_merge_0.reset_index(drop=True)
            k_list.append(df_trade_merge)
        
        df_trade = pd.concat(k_list,axis=0)
        
        df_trade = df_trade.drop(['year_y'],axis=1).reset_index(drop=True).rename({'year_x': 'year'},axis=1)
    
        return df_trade
        

    """
    Generating variable  "demaut" from year = 1885  to year 2010
    """
    
    def gen_polity(self, data_dyad,data_polity):
        
        df_polity = pd.DataFrame({'year' : data_dyad.year, 'dyadid': data_dyad.dyadid,\
                                  'ccode1': data_dyad.ccode1, 'ccode2':data_dyad.ccode2})
        p_list = []
        p_years = pd.unique(data_dyad['year'])
        
        for iyear in range(len(p_years)):
            i_year = p_years[iyear]
            data_dyad_year = data_dyad.loc[(data_dyad.year == i_year),:].reset_index(drop=True)
            data_polity_year = data_polity.loc[(data_polity.year == i_year),:].reset_index(drop=True)
    
            m_list = []
            polity_ccode = data_polity_year.ccode
            for i in range(len(data_dyad_year)):
                i_dyadid = data_dyad_year.dyadid[i]
                ccode_a = data_dyad_year.ccode1[i]
                ccode_b = data_dyad_year.ccode2[i]
                if (np.isin(ccode_a,polity_ccode)) and (np.isin(ccode_b,polity_ccode)):
                    ccode_1 = pd.Index(data_polity_year['ccode']).get_loc(ccode_a)
                    ccode_2 = pd.Index(data_polity_year['ccode']).get_loc(ccode_b)
                    m_list.append(min(data_polity_year.demaut[ccode_1],data_polity_year.demaut[ccode_2]))
                else:
                    m_list.append(np.nan)
    
            p_list.append(pd.DataFrame({'smldmat': m_list}))
            
        df_polity['smldmat'] = (pd.concat(p_list,axis=0)).reset_index(drop=True)
        
        return df_polity
        

    """ 
    Generating variable "allies" from year = 1885 to year = 2010
    Computing allies = equals 1 if the dyad members were linked by 
    formal mutual defense treaties, neutrality pacts, nonaggression pacts, 
    or entente, and equals 0 otherwise
    """
    
    def gen_allies(self, df_dyad, data_allies_0):
    
        data_allies_1 = data_allies_0[['year','ccode1', 'ccode2', 'defense', 'neutrality', \
                        'nonaggression','entente']].sort_values(by = ['year','ccode1','ccode2'])
        data_allies_origin = (data_allies_1.loc[(data_allies_1['year'] > 1884) \
                                            & (data_allies_1['year'] < 2011) ,:]).reset_index(drop=True)
        data_allies_dyad = pd.DataFrame({'dyadid': 1000*data_allies_origin['ccode1'] \
                                         + data_allies_origin['ccode2']})
        data_allies_init = pd.DataFrame({'allies': np.zeros(len(data_allies_origin))})
        data_allies = pd.concat([data_allies_origin,data_allies_dyad,data_allies_init],axis=1)
        data_allies = data_allies[['year','dyadid','ccode1','ccode2','defense','neutrality',\
                                   'nonaggression','entente','allies']]
    
        data_allies.loc[data_allies['defense'] == 1, 'allies'] = 1
        data_allies.loc[data_allies['neutrality'] == 1, 'allies'] = 1
        data_allies.loc[data_allies['nonaggression'] == 1, 'allies'] = 1
        data_allies.loc[data_allies['entente'] == 1, 'allies'] = 1
    
        data_allies = data_allies[['year', 'ccode1', 'ccode2','dyadid', 'defense', 'neutrality',
               'nonaggression', 'entente', 'allies']].sort_values(by = ['year','dyadid'])
        data_allies = data_allies.drop(columns = ['defense','neutrality','nonaggression','entente'])
    
        df_allies =  pd.DataFrame({'year': df_dyad.year, 'dyadid': df_dyad.dyadid, 'allies': \
                                   np.zeros(len(df_dyad))})
    
        a_years = pd.unique(df_allies['year'])
        
        df_list = []
        for iyear in range(len(a_years)):
            df_allies_year = df_allies.loc[df_allies.year == a_years[iyear],:].reset_index(drop=True)
            data_allies_year = data_allies.loc[data_allies.year == a_years[iyear],:].reset_index(drop=True)
            df_common_dyads = np.intersect1d(df_allies_year['dyadid'],data_allies_year['dyadid'])
            df_allies_year.loc[df_allies_year['dyadid'].isin(df_common_dyads),'allies'] = 1
            df_list.append(df_allies_year)
    
        df_allies = (pd.concat(df_list,axis=0).sort_values(by = ['year','dyadid'])).reset_index(drop=True)
    
        return df_allies


    """ 
    Generatin variable "log(relative capabilities ratio)" for each dyad 
    from year = 1885 to year = 2010
    """
    
    def gen_lcaprat(self, data_dyad, data_lcap):
        
        a_list = []
        a_years = pd.unique(data_lcap['year'])
        for iyear in range(len(a_years)):
            data_dyad_year = data_dyad.loc[data_dyad.year == a_years[iyear],:].reset_index(drop=True)
            data_lcap_year = data_lcap.loc[data_lcap.year == a_years[iyear],:].reset_index(drop=True)
            
            b_list = []
            for idyad in range(len(data_dyad_year)):
                ccode_1 = data_dyad_year.ccode1[idyad]
                ccode_2 = data_dyad_year.ccode2[idyad]
                ccode_a = pd.Index(data_lcap_year['ccode']).get_loc(ccode_1)
                ccode_b = pd.Index(data_lcap_year['ccode']).get_loc(ccode_2)
                b_list.append(data_lcap_year.lcap[ccode_a] - data_lcap_year.lcap[ccode_b])
            
            df_a_list = pd.DataFrame(b_list,columns=['lcaprat'])
            df_a = pd.concat([data_dyad_year,df_a_list], axis =1)
            a_list.append(df_a)
        
        df_lcaprat = pd.concat(a_list,axis=0).reset_index(drop=True)
    
        return df_lcaprat
        

    """
    Generatin variable for directed contiguity "contigkb" from year = 1885 to year = 2010
    Transformation of "contigkb" (1,2,3,4,5) to 0 if two states are not directly or 
    indirectly contiguous and 1 if they share a territorial boundary or are 
    divided by less than 150 miles of water (1,2,3,4) => 1 and 5 => 0
    """

    def gen_contgkf(self, data_dyad, data_contgkf):
    
        data_contgkf.loc[(data_contgkf['contgkf'] == 5),'contgkf'] = 0
        for i in range(4):
            data_contgkf.loc[(data_contgkf['contgkf'] == i + 1),'contgkf'] = 1
        data_contgkf = data_contgkf[['year','dyadid','ccode1','ccode2','contgkf']]
        a_years = pd.unique(data_contgkf['year'])
        c_list = []
        for iyear in range(len(a_years)):
            data_dyad_year = data_dyad.loc[data_dyad.year == a_years[iyear],:].reset_index(drop=True)
            df_common_dyads = pd.DataFrame({'dyadid': 
                                            np.intersect1d(data_dyad_year.dyadid,data_contgkf.dyadid)})
            
            data_contgkf_year = (data_contgkf.loc[data_contgkf['dyadid'].isin(df_common_dyads['dyadid']),
                                                  :]).reset_index(drop=True)
    
            ck_list, year_list, dyadid_list = [], [] , []
            for idyad in range(len(data_contgkf_year)):
                dyadid_value = data_contgkf_year.dyadid[idyad]
                ck_list.append(data_contgkf_year.contgkf[idyad])
                year_list.append(a_years[iyear])
                dyadid_list.append(dyadid_value)
    
            df_m = pd.DataFrame({'year': year_list, 'dyadid': dyadid_list, 'contgkf': ck_list})
            df_m.drop_duplicates(subset=['year','dyadid'],ignore_index=True,inplace=True)
            df_merge = pd.merge(left = data_dyad_year,right = df_m, how ='outer', on = ['dyadid'] )
            c_list.append(df_merge)
    
        df_contgkf = pd.concat(c_list,axis=0).reset_index(drop=True).drop(['year_y'],\
                                                    axis=1).rename(columns={'year_x': 'year'})
        df_contgkf.fillna(value = 0,inplace = True)
    
        return df_contgkf
        

    """ 
    Generating variable hegdef : dyadic ratio of military 
    expenditures: abs(mil_1 - mil_2)/sum(mil_1 + mil_2)
    """
    
    def gen_hegdef(self, data_dyad,data_hegdef):
    
        a_list = []
        a_years = pd.unique(data_hegdef['year'])
        for iyear in range(len(a_years)):
            data_dyad_year = data_dyad.loc[data_dyad.year == a_years[iyear],:].reset_index(drop=True)
            data_hegdef_year = data_hegdef.loc[data_hegdef.year == a_years[iyear],:].reset_index(drop=True)
            
            mil_1_list, mil_2_list, b_list = [], [] ,[]
            
            for idyad in range(len(data_dyad_year)):
                ccode_1 = data_dyad_year.ccode1[idyad]
                ccode_2 = data_dyad_year.ccode2[idyad]
    
                if (np.isin(ccode_1,data_hegdef_year.ccode)) and (np.isin(ccode_2,data_hegdef_year.ccode)):
                    ccode_a = pd.Index(data_hegdef_year['ccode']).get_loc(ccode_1)
                    ccode_b = pd.Index(data_hegdef_year['ccode']).get_loc(ccode_2)
                    mil_1 = data_hegdef_year.milex[ccode_a]
                    mil_1_list.append(mil_1)
                    mil_2 = data_hegdef_year.milex[ccode_b]
                    mil_2_list.append(mil_2)
                    b_list.append(abs((mil_1 - mil_2)/(mil_1 + mil_2)))
                else:
                    mil_1_list.append(np.nan)
                    mil_2_list.append(np.nan)
                    b_list.append(np.nan)
    
            df_a_list = pd.DataFrame({'mil_1': mil_1_list, 'mil_2': mil_2_list, 'hegdef': b_list})
            df_a = pd.concat([data_dyad_year,df_a_list], axis =1)
            hegdef_max = df_a['hegdef'].max(skipna=True)
            df_a['hegdef'] = df_a['hegdef']/hegdef_max
            a_list.append(df_a)
    
        df_hegdef = pd.concat(a_list,axis=0).reset_index(drop=True)
    
        return df_hegdef

    """ 
    GDP per capita ppp inflation adjusted
    """
    
    def gen_data_gdp(self):
        
        folder = self.folder
        
        data_states = pd.read_csv(folder+'COW country codes.csv')
    
        data_gdp = pd.read_csv(folder+'gdppercapita.csv',)
    
        df_country = pd.DataFrame({'statename': data_gdp.country})
        states_list_0 =  pd.merge(left = df_country ,right = data_states, how ='inner', 
                                  on = ['statename'] )
        states_list = states_list_0.drop_duplicates().reset_index(drop=True)
        
        year_list = len(data_gdp.columns.values[1:])
    
        df_data = data_gdp.iloc[0:,1:]
    
        a_list = []
        for i in range(year_list):
            
            ay_list, ac_list, b_list = [], [], []
            for j in range(len(states_list)):
                b_list.append(df_data.iloc[j,i])
                ay_list.append(1800+i)
                ac_list.append(states_list.ccode[j])
            
            a_list.append(pd.DataFrame({'year': ay_list, 'ccode': ac_list, 'gdp': b_list}))
        
        df_a = pd.concat(a_list, axis = 0)
    
        df_gdp = df_a.loc[(df_a['year'] >= 1885) & (df_a['year'] <= 2010),:].sort_values(by 
                                                                                         =['year','ccode'])
    
        return df_gdb


    """ 
    Generating variable logdstab (log of distance between capitals of states in dyad)
    """
    
    def gen_lgmidist(self, data_dyad, data_midist):
    
        a_list = []
        a_years = pd.unique(data_dyad['year'])
        for iyear in range(len(a_years)):
    
            data_dyad_year = data_dyad.loc[data_dyad.year == a_years[iyear],:].reset_index(drop=True)
            df_common_dyads = pd.DataFrame({'dyadid': np.intersect1d(data_dyad_year.dyadid,data_midist.dyadid)})
            data_midist_year = (data_midist.loc[data_midist['dyadid'].isin(df_common_dyads['dyadid']),\
                                                :]).reset_index(drop=True)
            
            lgm_value_list, year_list, dyadid_list = [], [] ,[]
            for idyad in range(len(data_midist_year)):
                dyadid_value = data_midist_year.dyadid[idyad]
                lgm_value_list.append(np.log(data_midist_year.midist[idyad]))
                year_list.append(a_years[iyear])
                dyadid_list.append(dyadid_value)
    
            df_m = pd.DataFrame({'year': year_list, 'dyadid': dyadid_list, 'lgmidist': lgm_value_list})
            df_m.drop_duplicates(subset=['year','dyadid'],ignore_index=True,inplace=True)
            df_merge = pd.merge(left = data_dyad_year,right = df_m, how ='outer', on = 'dyadid' )
            a_list.append(df_merge)
    
        df_lgmidist = pd.concat(a_list,axis=0).reset_index(drop=True).drop(['year_y'],\
                                                    axis=1).rename(columns={'year_x': 'year'})
    
    
        return df_lgmidist
        

    """ 
    Generating  variable disp and disp_lag on MID 
    """
    
    def gen_disp(self, data_dyad,data_disp):
        df_disp = pd.DataFrame({'year': data_dyad.year, 'dyadid': data_dyad.dyadid, \
                                                                'disp': np.zeros(len(data_dyad))})
        n_years = (pd.DataFrame({'year': pd.unique(data_dyad['year'])})).sort_values(by = ['year'])
        disp_years = pd.DataFrame({'year': pd.unique(data_disp['year'])}).sort_values(by = ['year'])
    
        disp_list = []
        for iyear in range(len(n_years)):
            df_disp_year = df_disp.loc[df_disp.year \
                                        == n_years.year[iyear],:].reset_index(drop=True)
            data_dyad_year = data_dyad.loc[data_dyad.year \
                                        == n_years.year[iyear],:].reset_index(drop=True)
            data_disp_year = data_disp.loc[data_disp.year \
                                        == n_years.year[iyear],:].reset_index(drop=True)
            common_dyads = np.intersect1d(data_dyad_year['dyadid'],data_disp_year['dyadid'])
            df_disp_year.loc[df_disp_year['dyadid'].isin(common_dyads),'disp'] = 1
            disp_list.append(df_disp_year)
    
        df_disp = (pd.concat(disp_list,axis=0)).reset_index(drop=True)
        df_disp_lag = pd.DataFrame({'disp_lag': df_disp['disp'].copy().shift(fill_value=0)}).\
                                reset_index(drop=True)
        
        return [df_disp,df_disp_lag]
        

    """ 
    Loading data from year = 1885 to year = 2010
    """
    
    def reading_all_input_data(self):
        folder = self.folder
        
        """ States in the system of states per year """
    
        file_system ='system2016.csv'
        data_system_all = pd.read_csv(folder+file_system,usecols=['stateabb', 'ccode', 'year'])
        data_system = data_system_all.loc[(data_system_all.year > 1884) \
                                          & (data_system_all.year < 2011),:].reset_index(drop=True)
        data_system.to_csv(folder+'data_system.csv')
    
        """ Data on dyads """
    
        file_dyad = 'dyadic_format_v3.dta'
        data_dyad = pd.read_stata(folder+file_dyad, columns = ['year','ccode1','ccode2'])
    
        """ Data on disp - (MID) military International disputes from year = 1885 to year = 2010 """
    
        file_disp = 'Dyadic MID Dataset_short.csv'
        data_disp_0 = pd.read_csv(folder+file_disp, usecols = ['year','dyadid','ccode1','ccode2','disp'])
        data_disp_1 = data_disp_0[['year','dyadid','ccode1','ccode2','disp']].sort_values(by = ['year','ccode1'])
        data_disp = data_disp_1.loc[(data_disp_1['year']> 1884) & (data_disp_1['year'] < 2011),:].reset_index(drop = True)
        data_disp.to_csv(folder+'data_disp.csv')
        
        """ Data on trade """
    
        file_trade = 'dyadic_trade_COW_4.0.csv'
        data_trade_0 = pd.read_csv(folder+file_trade, usecols = ['year','ccode1','ccode2','flow1','flow2'])
        data_trade_0['dyadid']=1000*data_trade_0['ccode1'] + data_trade_0['ccode2']
        data_trade_1 = data_trade_0[['year','dyadid','ccode1','ccode2','flow1', \
                                     'flow2']].sort_values(by = ['year','dyadid','ccode1'])
        data_trade_2 = data_trade_1.loc[(data_trade_1['year'] > 1884) & \
                                        (data_trade_1['year'] < 2011),:].reset_index(drop=True)
        indexNames_1 = data_trade_2[data_trade_2['flow1'] == 0].index
        data_trade_2.drop(indexNames_1, inplace=True)
        indexNames_2 = data_trade_2[ data_trade_2['flow2'] == 0 ].index
        data_trade_2.drop(indexNames_2,inplace=True)
        indexNames_3 = data_trade_2[ data_trade_2['flow2'] == -9 ].index
        data_trade_2.drop(indexNames_3,inplace=True)
        data_trade = data_trade_2.reset_index(drop=True)
        data_trade.to_csv(folder+'data_trade.csv')
    
        """ Data on polity (regime) scores """
    
        file_polity = 'p5v2018.csv'
        data_polity_00 = pd.read_csv(folder+file_polity, usecols = ['year','ccode','polity2'])
        data_polity_0 = data_polity_00.loc[data_polity_00.polity2 != -66,:].dropna().reset_index(drop=True)
        data_polity_1 = data_polity_0.rename(columns={'polity2':'demaut'}).sort_values(by = ['year','ccode'])
        data_polity_2 = data_polity_1.loc[(data_polity_1['year'] > 1884) & \
                                          (data_polity_1['year'] < 2011),:].reset_index(drop=True)
        data_polity = data_polity_2[['year','ccode','demaut']].reset_index(drop=True)
        data_polity.to_csv(folder+'data_polity.csv')
    
        """ Data on allies """
    
        file_allies = 'alliance_v4.1_by_dyad_yearly.csv'
        data_allies = pd.read_csv(folder+file_allies,usecols = ['ccode1', 'ccode2', 'year', 'defense', \
            'neutrality', 'nonaggression','entente'])
        data_allies.to_csv(folder+'data_allies.csv')
    
        """ Data on states capabilities """
    
        file_cinc = 'NMC_5_0.csv'
        data_lcap_0 = pd.read_csv(folder+file_cinc,usecols = ['ccode','year','cinc'])
        data_lcap = data_lcap_0[['year','ccode','cinc']].sort_values(by = ['year','ccode','cinc'])
        data_lcap.drop(data_lcap[(data_lcap['year'] < 1885)].index, inplace = True)
        data_lcap.drop(data_lcap[(data_lcap['year'] > 2010)].index, inplace = True)
        data_lcap['lcap'] = pd.DataFrame({'logcap': np.log(data_lcap.cinc)})
        data_lcap.to_csv(folder+'data_lacp.csv')
    
        """ Data on contiguity """
    
        file_contgkf = 'contdird.csv'
        data_contgkf_0 = pd.read_csv(folder+file_contgkf, usecols = ['dyad','state1no', \
                                 'state2no', 'year','conttype']).rename(columns={'state1no': \
                                'ccode1', 'state2no': 'ccode2','conttype': 'contgkf','dyad':'dyadid'})
        data_contgkf_1 = data_contgkf_0.loc[(data_contgkf_0['year'] >= 1885 ) \
                                            & (data_contgkf_0['year'] <= 2010),:]
        data_contgkf_2 = data_contgkf_1.sort_values(by = ['year','dyadid','ccode1',\
                                                          'ccode2']).reset_index(drop=True)
        data_contgkf = data_contgkf_2.loc[data_contgkf_2['ccode1'] < \
                                          data_contgkf_2['ccode2'],:].reset_index(drop=True)
        data_contgkf.to_csv(folder+'data_contgkf.csv')
    
        """ Data on hegemon within dyad """

        file_cinc = 'NMC_5_0.csv'
        data_hegdef_0 = pd.read_csv(folder+file_cinc,usecols = ['ccode','year','milex'])
        data_hegdef_1 = data_hegdef_0[['year','ccode','milex']].sort_values(by = ['year','ccode','milex'])
        data_hegdef_2 = data_hegdef_1.loc[(data_hegdef_1['year'] > 1884)\
                                          & (data_hegdef_1['year'] < 2011),:].reset_index(drop=True)
        data_hegdef = data_hegdef_2[['year','ccode','milex']]
        # Dropping null and missing values of milex
        data_hegdef = data_hegdef.loc[data_hegdef['milex']!=0,:]
        data_hegdef = data_hegdef.loc[data_hegdef['milex']!=-9,:]
        data_hegdef.to_csv(folder+'data_hegdef.csv')
    
        """ Data on distance between states in dyad """
    
        file_capdist = 'capdist.csv'
        data_midist_0 = pd.read_csv(folder+file_capdist,usecols = \
                                    ['dyadid','ccode1','statea','ccode2', 'stateb','midist'])
        
        """ Dropping dulicated dyads """
        data_midist = data_midist_0.loc[data_midist_0['ccode1']\
                                        < data_midist_0['ccode2'],:].reset_index(drop=True)
        
        """ Dropping null and missing values of midist """
        
        data_midist = data_midist.loc[data_midist['midist']!= 0,:]
        data_midist = data_midist.loc[data_midist['midist']!= -9,:]
        data_midist.to_csv(folder+'data_midist.csv')
    
        data_list = [data_dyad,data_disp,data_trade,data_polity,
                     data_allies,data_lcap,data_contgkf,data_hegdef,data_midist]
    
        return data_list
        

    """
    Calculating variables 
    """ 

    def gen_all_vars(self, data):
    
        df_dyad =               self.gen_dyad(data[0])
        [df_disp,df_disp_lag] = self.gen_disp(df_dyad,data[1])
        df_majpower =           self.gen_majpower(df_dyad)
        df_trade =              self.gen_depend(df_dyad,data[2])
        df_polity =             self.gen_polity(df_dyad,data[3])
        df_allies =             self.gen_allies(df_dyad,data[4])
        df_lcaprat =            self.gen_lcaprat(df_dyad,data[5])
        df_contgkf =            self.gen_contgkf(df_dyad,data[6])
        df_hegdef =             self.gen_hegdef(df_dyad,data[7])
        df_lgmidist =           self.gen_lgmidist(df_dyad,data[8])
    
        df_vars = pd.DataFrame({'disp': df_disp.disp, 
                                'disp_lag': df_disp_lag.disp_lag, 
                                'majpower': df_majpower.majpower,  
                                'smldep': df_trade.smldep,   
                                'smldmat': df_polity.smldmat,  
                                'allies': df_allies.allies, 
                                'lcaprat': df_lcaprat.lcaprat,  
                                'contgkf': df_contgkf.contgkf,  
                                'hegdef': df_hegdef.hegdef, 
                                'lgmidist': df_lgmidist.lgmidist 
                                })
    
        df_all = pd.concat([df_dyad,df_vars],axis=1)
    
        return df_all
    
    
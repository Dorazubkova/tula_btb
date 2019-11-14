#!/usr/bin/env python
# coding: utf-8

# In[1]:


import bokeh
from bokeh.server.server import Server as server
from bokeh.io import show, output_notebook
from bokeh.plotting import figure, show, output_notebook
from bokeh.tile_providers import Vendors, get_provider
import pandas as pd
import numpy as np
from bokeh.models import ColumnDataSource, HoverTool, LassoSelectTool, Label, Title, ZoomInTool, ZoomOutTool
from bokeh.layouts import gridplot
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.plotting import ColumnDataSource, Figure
from bokeh.models.widgets import PreText, Paragraph, Select, Dropdown, RadioButtonGroup, RangeSlider, Slider, CheckboxGroup,HTMLTemplateFormatter,TableColumn, RadioGroup
import bokeh.layouts as layout
from bokeh.models.widgets import Tabs, Panel
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import shapely 
from shapely.geometry import Point
import geopandas as gpd
tile_provider = get_provider(Vendors.CARTODBPOSITRON)


# In[2]:


mylistI = ['1','2','3']
mylistJ = ['0','1','2']
for i in mylistI:
    for j in mylistJ:
        exec(f"matrix_{i}_{j} = pd.read_csv(r'myapptula2/scenario_{i}_btb_matrix_transf_{j}.csv'.format(i, j), sep = ';', encoding='cp1251')")
        exec(f"matrix_{i}_{j}['Scenario'] = i")


# In[3]:


matrix_0 = pd.DataFrame()
matrix_0 = matrix_0.append(matrix_1_0)
matrix_0 = pd.merge(matrix_0,matrix_2_0, how='outer', on=['site_id_from', 'site_id_to'] )
matrix_0 = matrix_0[['site_id_from','site_id_to','cost_x','cost_y']].rename(columns={'cost_x':'cost_1', 'cost_y':'cost_2'})
matrix_0 = pd.merge(matrix_0,matrix_3_0, how='outer', on=['site_id_from', 'site_id_to'] )
matrix_0 = matrix_0[['site_id_from','site_id_to','cost_1','cost_2','cost']].rename(columns={'cost':'cost_3'})
matrix_0 = matrix_0.fillna(0)


# In[4]:


matrix_0.head()


# In[5]:


matrix_1 = pd.DataFrame()
matrix_1 = matrix_1.append(matrix_1_1)
matrix_1 = pd.merge(matrix_1,matrix_2_1, how='outer', on=['site_id_from', 'site_id_to'] )
matrix_1 = matrix_1[['site_id_from','site_id_to','cost_x','cost_y']].rename(columns={'cost_x':'cost_1', 'cost_y':'cost_2'})
matrix_1 = pd.merge(matrix_1,matrix_3_1, how='outer', on=['site_id_from', 'site_id_to'] )
matrix_1 = matrix_1[['site_id_from','site_id_to','cost_1','cost_2','cost']].rename(columns={'cost':'cost_3'})
matrix_1 = matrix_1.fillna(0)


# In[6]:


matrix_1.head()


# In[7]:


matrix_2 = pd.DataFrame()
matrix_2 = matrix_2.append(matrix_1_2)
matrix_2 = pd.merge(matrix_2,matrix_2_2, how='outer', on=['site_id_from', 'site_id_to'] )
matrix_2 = matrix_2[['site_id_from','site_id_to','cost_x','cost_y']].rename(columns={'cost_x':'cost_1', 'cost_y':'cost_2'})
matrix_2 = pd.merge(matrix_2,matrix_3_2, how='outer', on=['site_id_from', 'site_id_to'] )
matrix_2 = matrix_2[['site_id_from','site_id_to','cost_1','cost_2','cost']].rename(columns={'cost':'cost_3'})
matrix_2 = matrix_2.fillna(0)


# In[8]:


matrix_2.head()


# In[9]:


sites_centr = pd.read_csv('myapptula2/sites_centr.csv', sep = ';', encoding='cp1251')
sites_centr.head()


# In[ ]:


cds = dict(X=list(sites_centr['X'].values), 
                    Y=list(sites_centr['Y'].values),
                    site_id=list(sites_centr['site_id'].values))


# In[ ]:


source_from = ColumnDataSource(data = cds)
source_to = ColumnDataSource(data = cds)
source_from2 = ColumnDataSource(data = cds)
source_to2 = ColumnDataSource(data = cds)


# In[10]:


for j in mylistJ:
    exec(f"matrix_{j} = pd.merge(matrix_{j}, sites_centr, how = 'inner', left_on = ['site_id_from'], right_on = ['site_id'])")        
    exec(f"matrix_{j} = matrix_{j}[['site_id_from','site_id_to','cost_1', 'cost_2','cost_3','X','Y']]")
    exec(f"matrix_{j}.columns = ['site_id_from','site_id_to','cost_1', 'cost_2','cost_3','X_from','Y_from']")

    exec(f"matrix_{j} = pd.merge(matrix_{j}, sites_centr, how = 'inner', left_on = ['site_id_to'], right_on = ['site_id'])")        
    exec(f"matrix_{j} = matrix_{j}[['site_id_from','site_id_to','cost_1', 'cost_2','cost_3','X_from','Y_from','X','Y']]")
    exec(f"matrix_{j}.columns = ['site_id_from','site_id_to','cost_1', 'cost_2','cost_3','X_from','Y_from','X_to','Y_to']")       


# In[11]:


matrix_0.head()


# In[12]:


cds_empty = dict(
                X_from = [],
                Y_from = [],
                X_to = [],
                Y_to = [],
                cost_1=[],
                cost_2=[],
                cost_3=[],
                           )


# In[13]:


# mylistI2 = ['1','2','3']
# mylistJ2 = ['0','1','2']

# for i in mylistI2:
#     for j in mylistJ2:
#         exec(f'source_from = ColumnDataSource(data = cds)')
#         exec(f'source_to = ColumnDataSource(data = cds)')
#         exec(f'source_from2 = ColumnDataSource(data = cds)')
#         exec(f'source_to2 = ColumnDataSource(data = cds)')


# In[14]:


lasso_from = LassoSelectTool(select_every_mousemove=False)
lasso_to = LassoSelectTool(select_every_mousemove=False)

lasso_from2 = LassoSelectTool(select_every_mousemove=False)
lasso_to2 = LassoSelectTool(select_every_mousemove=False)

toolList_from = [lasso_from,  'reset',  'pan','wheel_zoom']
toolList_to = [lasso_to,  'reset',  'pan', 'wheel_zoom']

toolList_from2 = [lasso_from2, 'reset', 'pan','wheel_zoom']
toolList_to2 = [lasso_to2,  'reset',  'pan','wheel_zoom']


# In[15]:


#рисуем графики
p = figure(x_range=(4155911, 4206523), y_range=(7185880, 7226515), x_axis_type="mercator", y_axis_type="mercator", tools=toolList_from)
p.add_tile(tile_provider)

#слой сайтов from
r = p.circle(x = 'X',
         y = 'Y',
         source=source_from,
        fill_color='navy',
        size=10,
        fill_alpha = 1,
        nonselection_fill_alpha=1,
        nonselection_fill_color='gray')

Time_Title1 = Title(text='Матрица: ', text_font_size='10pt', text_color = 'grey')
p.add_layout(Time_Title1, 'above')


# In[16]:


p_to = figure(x_range=(4155911, 4206523), y_range=(7185880, 7226515), x_axis_type="mercator", y_axis_type="mercator", tools=toolList_to)
p_to.add_tile(tile_provider)

t = p_to.circle(x = 'X', 
                y = 'Y', 
                fill_color='papayawhip', 
                fill_alpha = 0.6, 
                line_color='tan', 
                line_alpha = 0.8, 
                size=6 , 
                source = source_to,
                   nonselection_fill_alpha = 0.6, 
                nonselection_fill_color = 'papayawhip', 
                nonselection_line_color = None)

t_to1 = p_to.circle(x = [], y = [], fill_color=[], fill_alpha = 0.6, 
                    line_color= None, line_alpha = 0.8, size=20, nonselection_line_color = None,
                   nonselection_fill_color = 'papayawhip') 
t_to2 = p_to.diamond(x = [], y = [], fill_color=[], fill_alpha = 0.6, 
                    line_color= None, line_alpha = 0.8, size=20, nonselection_line_color = None,
                   nonselection_fill_color = 'papayawhip') 
t_to3 = p_to.hex(x = [], y = [], fill_color=[], fill_alpha = 0.6, 
                    line_color= None, line_alpha = 0.8, size=20, nonselection_line_color = None,
                   nonselection_fill_color = 'papayawhip') 


ds = r.data_source
tds = t.data_source
tds_to1 = t_to1.data_source
tds_to2 = t_to2.data_source
tds_to3 = t_to3.data_source


# In[17]:


checkbox_group1 = CheckboxGroup(labels=["1 сценарий", "2 сценарий", "3 сценарий"], active=[]) #можно несколько
radio_group1 = RadioGroup(
        labels=['0 пересадок','1 пересадка','2 пересадки'], active=0) #только 1 вариант


# In[18]:


matrix_0.head()


# In[ ]:


def callback(attrname, old, new): 
    
    idx = source_from.selected.indices
    print(idx)
    
    per = radio_group1.active
    print(per)
    
    sc = checkbox_group1.active
    print(sc)
    
    if per == 0:
        tbl = matrix_0
    elif per == 1:
        tbl = matrix_1
    else:
        tbl = matrix_2

    #таблица с выбранными индексами 
    df = pd.DataFrame(data=ds.data).iloc[idx]
    print(df)
    
    df1 = pd.merge(df, tbl, how = 'inner', left_on = ['site_id'], right_on = ['site_id_from'])
    print(df1)
    print(df1['cost_1'])
    
    
    color_1 = [ ]    
    for i in range(len(df1)):
        
        if 0 < df1['cost_1'][i] <= 5:
            color_1.append('darkblue')
        elif 5 < df1['cost_1'][i] <= 10:
            color_1.append('blue')
        elif 10 < df1['cost_1'][i] <= 15:
            color_1.append('aqua')
        elif 15 < df1['cost_1'][i] <= 30:
            color_1.append('green')
        elif 30 < df1['cost_1'][i] <= 50:
            color_1.append('greenyellow')
        elif 50 < df1['cost_1'][i] <= 75:
            color_1.append('yellow')
        elif 75 < df1['cost_1'][i] <= 100:
            color_1.append('orange')
        elif 100 < df1['cost_1'][i] <= 125:
            color_1.append('red')
        elif 125 < df1['cost_1'][i]:
            color_1.append('darkred')
        elif df1['cost_1'][i] == 0:
            color_1.append('None')
            
    color_2 = [ ]    
    for i in range(len(df1)):
        
        if 0 < df1['cost_2'][i] <= 5:
            color_2.append('darkblue')
        elif 5 < df1['cost_2'][i] <= 10:
            color_2.append('blue')
        elif 10 < df1['cost_2'][i] <= 15:
            color_2.append('aqua')
        elif 15 < df1['cost_2'][i] <= 30:
            color_2.append('green')
        elif 30 < df1['cost_2'][i] <= 50:
            color_2.append('greenyellow')
        elif 50 < df1['cost_2'][i] <= 75:
            color_2.append('yellow')
        elif 75 < df1['cost_2'][i] <= 100:
            color_2.append('orange')
        elif 100 < df1['cost_2'][i] <= 125:
            color_2.append('red')
        elif 125 < df1['cost_2'][i]:
            color_2.append('darkred')
        elif df1['cost_2'][i] == 0:
            color_2.append('None')
            
    color_3 = [ ]    
    for i in range(len(df1)):
        
        if 0 < df1['cost_3'][i] <= 5:
            color_3.append('darkblue')
        elif 5 < df1['cost_3'][i] <= 10:
            color_3.append('blue')
        elif 10 < df1['cost_3'][i] <= 15:
            color_3.append('aqua')
        elif 15 < df1['cost_3'][i] <= 30:
            color_3.append('green')
        elif 30 < df1['cost_3'][i] <= 50:
            color_3.append('greenyellow')
        elif 50 < df1['cost_3'][i] <= 75:
            color_3.append('yellow')
        elif 75 < df1['cost_3'][i] <= 100:
            color_3.append('orange')
        elif 100 < df1['cost_3'][i] <= 125:
            color_3.append('red')
        elif 125 < df1['cost_3'][i]:
            color_3.append('darkred')
        elif df1['cost_3'][i] == 0:
            color_3.append('None')
            
    print(color_1)
    print(color_2)
    print(color_3)
    
    new_data1 = dict()
    new_data2 = dict()
    new_data3 = dict()
    
    if 0 in sc:  

        new_data1['x'] = list(df1['X_to'])
        new_data1['y'] = list(df1['Y_to'])
        new_data1['fill_color'] = color_1
        tds_to1.data = new_data1
        
    else:
        
        new_data1['x'] = []
        new_data1['y'] = []
        new_data1['fill_color'] = []
        tds_to1.data = new_data1
        
    
    if 1 in sc: 
    
        new_data2['x'] = list(df1['X_to'])
        new_data2['y'] = list(df1['Y_to'])
        new_data2['fill_color'] = color_2
        tds_to2.data = new_data2
        
    else:
        
        new_data2['x'] = []
        new_data2['y'] = []
        new_data2['fill_color'] = []
        tds_to2.data = new_data2
        
    if 2 in sc: 

        new_data3['x'] = list(df1['X_to'])
        new_data3['y'] = list(df1['Y_to'])
        new_data3['fill_color'] = color_3
        tds_to3.data = new_data3
        
    else:
        
        new_data3['x'] = []
        new_data3['y'] = []
        new_data3['fill_color'] = []
        tds_to3.data = new_data3

    
source_from.selected.on_change('indices', callback)
radio_group1.on_change('active', callback)
checkbox_group1.on_change('active', callback)


# In[19]:


# def update1(attrname, old, new):
    
#     per = radio_group1.active
#     print(per)
    
#     if per == 0:
#         df = matrix_0
#     elif per == 1:
#         df = matrix_1
#     else:
#         df = matrix_2
    
#     print(df)
    
#     cds_upd1 = dict(X_from=list(df['X_from'].values), 
#                     Y_from=list(df['Y_from'].values),
#                     X_to=list(df['X_to'].values), 
#                     Y_to=list(df['Y_to'].values),
#                     cost_1=list(df['cost_1'].values),
#                     cost_2=list(df['cost_2'].values),
#                     cost_3=list(df['cost_3'].values))

#     source_from.data = ColumnDataSource(data = cds_upd1).data
#     source_to.data = ColumnDataSource(data = cds_upd1).data

        
#         exec(f"cds_upd1_{j+1}_{per} = dict( \
#              X_from=list(matrix_{j+1}_{per}['X_from'].values), \
#              Y_from=list(matrix_{j+1}_{per}['Y_from'].values), \
#              X_to=list(matrix_{j+1}_{per}['X_to'].values), \
#              Y_to=list(matrix_{j+1}_{per}['Y_to'].values), \
#              cost=list(matrix_{j+1}_{per}['cost'].values) \
#                 )")

#         exec(f"source_from_{j+1}_{per}.data = ColumnDataSource(data = cds_upd1_{j+1}_{per}).data") 
#         exec(f"source_to_{j+1}_{per}.data = ColumnDataSource(data = cds_upd1_{j+1}_{per}).data") 
#         exec(f"source_from2_{j+1}_{per}.data = ColumnDataSource(data = cds_upd1_{j+1}_{per}).data")
#         exec(f"source_to2_{j+1}_{per}.data = ColumnDataSource(data = cds_upd1_{j+1}_{per}).data")

    
    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[23]:


layout1 = layout.row(p,p_to)
layout2 = layout.row(radio_group1, checkbox_group1)
layout3 = layout.row(layout1, layout2)

tab1 = Panel(child=layout3)

tabs = Tabs(tabs=[tab1])

doc = curdoc() #.add_root(tabs)
#doc.theme = theme
doc.add_root(tabs)


# In[ ]:





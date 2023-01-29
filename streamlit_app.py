###############################################################################################################
# App de explicacion como funciona Algoritmo kmeans
###############################################################################################################


#**************************************************************************************************************
# [A] Importar LIbrerias a Utilizar
#**************************************************************************************************************

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans

import streamlit as st

#**************************************************************************************************************
# [B] Crear funciones utiles para posterior uso
#**************************************************************************************************************

# definir funcion de distancia (en este caso euclidiana)
def dist_e(x1,y1,x2,y2):
  return ((y2-y1)**2+(x2-x1)**2)**0.5


#**************************************************************************************************************
# [Y] Comenzar a diseñar App
#**************************************************************************************************************

def main():
  
  # Use the full page instead of a narrow central column
  st.set_page_config(layout="wide")

  #=============================================================================================================
  # [01] Elementos del sidebar: textos 
  #=============================================================================================================   

  # autoria 
  st.sidebar.markdown('**Autor: [Sebastian Barrera](https://www.linkedin.com/in/sebasti%C3%A1n-nicolas-barrera-varas-70699a28)**')
  st.sidebar.markdown('Version SB_V20230128')
  
  # ingresar parametro de cantidad de nubes
  n_nubes = st.sidebar.number_input(
    label = 'cantidad de nubes a generar:',
    min_value=1,
    max_value=8,
    value=2,
    step=1    
    )
  
  # crear arreglo de valores  
  num_c = [None] * n_nubes
  num_s = [None] * n_nubes
  
  # ingresar valores de nubes 
  for i in range(n_nubes):
    
    col_1a, col_1b = st.sidebar.columns((2,1))
    
    if(i==0):
    
      num_c[i] = col_1a.text_input(
        label = 'centro (x,y)',
        value='('+str(11+i*11)+','+str(11+i*8)+')',
        key='num_c'+str(i)
        )
      
      num_s[i] = col_1b.text_input(
        label = 'desviacion',
        value=str(2.3+0.5*i),
        key='num_s'+str(i)
        )

    else:
      
      num_c[i] = col_1a.text_input(
        label = 'centro (x,y)',
        value='('+str(11+i*11)+','+str(11+i*8)+')',
        key='num_c'+str(i),
        label_visibility='collapsed'
        )
      
      num_s[i] = col_1b.text_input(
        label = 'desviacion',
        value=str(2.3+0.5*i),
        key='num_s'+str(i),
        label_visibility='collapsed'
        )
      
  # agregar boton de generar nubes
  boton_generar_nube = st.sidebar.button(label='generar nube')
  


  #=============================================================================================================
  # [02] Elementos del main: Titulo
  #=============================================================================================================   
      
  # titulo inicial 
  st.markdown('# Algoritmo de Clusterizacion k-means')
  st.markdown('---')
  
  #=============================================================================================================
  # [03] Elementos del main: Generar grafico de nube de puntos y grafico del codo
  #=============================================================================================================   
    
  if st.session_state.get('button') != True:

    st.session_state['button'] = boton_generar_nube
  
  
  if st.session_state['button']:
    
    
    #-----------------------------------------------------------------------------------------------------------
    # [03.1] crear funcion de entregables relativos a la nube de puntos 
    #-----------------------------------------------------------------------------------------------------------
    
    @st.cache(suppress_st_warning=True,allow_output_mutation=True) # https://docs.streamlit.io/library/advanced-features/caching
    def generar_entregables_nube(
      lista_num_c,
      lista_num_s,
      n_registros_x_nube
    ):
    
      # generar lista de centros 
      l_centros = [eval(x) for x in lista_num_c]
      l_desv = [float(x) for x in lista_num_s]
      
      # crear df e ir completando
      n_registros_x_nube = 200 
      df1 = pd.DataFrame([])

      for i in range(n_nubes):

        df1 = pd.concat([
          df1,
          pd.DataFrame({
            'Nube':'n'+str(i+1),
            'v1': np.random.normal(l_centros[i][0], l_desv[i], n_registros_x_nube),
            'v2': np.random.normal(l_centros[i][1], l_desv[i], n_registros_x_nube)
          })
        ],axis=0)
        
        
      # calcular limites de ejes para posterior graficos
      lim_max = max(max(df1['v1']),max(df1['v2']))
      lim_max = lim_max + 0.1*abs(lim_max)

      lim_min = min(min(df1['v1']),min(df1['v2']))
      lim_min = lim_min - 0.1*abs(lim_min)

      # ver graficamente los resultados
      fig_nube = go.Figure(go.Scatter(
        mode='markers', 
        x=df1['v1'], 
        y=df1['v2'], 
        marker_size=7,
        marker_color='gray'
      ))

      fig_nube.update_layout(
        title='Puntos a segmentar',
        xaxis_title='variable v1 (eje x)',
        yaxis_title='variable v2 (eje y)',
        width=400,height=400,
        xaxis_range=[lim_min,lim_max],
        yaxis_range=[lim_min,lim_max]
        )
      
      # definir parametros iniciales de max de clusters y vector de inercia donde guardar
      Inercia = [] 
      numK = 10
      
      # recorrer por numero de cluster y calcular inercia 
      for k in range(2, numK): 
          
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df1[['v1','v2']])
        Inercia.append(kmeans.inertia_)

      df_km = pd.DataFrame({
        'k': np.arange(2,numK),
        'Inercia': Inercia
      })

      # generar grafico 
      fig_km = px.line(df_km,x='k',y='Inercia')
      fig_km.update_layout(
        title='Grafico del "codo" para elegir mejor k (Inercia vs k)',
        width=400,height=400      
        )
      
      return fig_nube,fig_km,df1
      
    #-----------------------------------------------------------------------------------------------------------
    # [03.2] Generar grafico de dispersion de nubes y de kmeans
    #-----------------------------------------------------------------------------------------------------------
    
    fig_nube,fig_km,df1 = generar_entregables_nube(
      lista_num_c = num_c,
      lista_num_s = num_s,
      n_registros_x_nube=100
    )
    
    # subtitulo
    st.markdown('### A. Configuracion inicial')
    
    # separar elementos en columnas 
    col_2a, col_2b = st.columns((1,1))
    
    # mostrar grafico de nube  
    col_2a.plotly_chart(fig_nube, use_container_width=True)
    
    # mostrar grafico de regla del codo       
    col_2b.plotly_chart(fig_km, use_container_width=True)


  #=============================================================================================================
  # [04] Elementos del main: Parametros para iteracion kmeans
  #=============================================================================================================   
    
    # linea separadora 
    st.markdown('---')
    
    # subtitulo
    st.markdown('### B. Iteraciones del Algoritmo k-means')
    
    # separar elementos en columnas 
    col_3a, col_3b, col_3c,col_3d = st.columns((1,1,1,1))
    
    
    col_3a.write('Nro Clusters:')
    
    # numero de clusters 
    nk = col_3b.number_input(
      label = 'cantidad de clusters:',
      min_value=2,
      max_value=8,
      value=3,
      step=1,
      label_visibility='collapsed'
      )
    
    
    col_3c.write('   ')
    
    # boton de inicializar algoritmo 
    boton_generar_cluster = col_3d.button(label='Iterar')
    
        
    #-----------------------------------------------------------------------------------------------------------
    # [04.1] cuando se apreta boton de generar clusters
    #-----------------------------------------------------------------------------------------------------------

    if boton_generar_cluster:
    
      #__________________________________________________________________
      # crear funcion que retorna entregables de iteraciones kmeans
      
      @st.cache(suppress_st_warning=True,allow_output_mutation=True) # https://docs.streamlit.io/library/advanced-features/caching
      def generar_entregables_kmeans(
        max_iter,
        k,
        df1
      ):
        
        # crear lista donde ir respaldando graficos y centroides
        l_graficos=[]
        l_centroides=[]
        l_centroides2=[]
        
        # calcular limites de ejes para posterior graficos
        lim_max = max(max(df1['v1']),max(df1['v2']))
        lim_max = lim_max + 0.1*abs(lim_max)

        lim_min = min(min(df1['v1']),min(df1['v2']))
        lim_min = lim_min - 0.1*abs(lim_min)
                
        for q in range(max_iter):
          
          # crear df de centroides aleatorios inicial si es 1ra iteracion
          if(q==0):
            
            df2 = pd.DataFrame([])
            for i in range(k):

              df2 = pd.concat([
                df2,
                pd.DataFrame({
                  'Cluster':'k'+str(i+1),
                  'v1': [np.random.uniform(min(df1['v1']), max(df1['v1']))],
                  'v2': [np.random.uniform(min(df1['v2']), max(df1['v2']))]
                })
              ],axis=0)
            
          # respaldar df de centroides 
          l_centroides.append(df2)
          
          # calcular distancia a cada cluster
          df1_2 = df1.copy()

          for i in range(k):
            
            # rescatar valores de centroide cluster
            k_v1 = df2.iloc[i,1].item()
            k_v2 = df2.iloc[i,2].item()
            
            df1_2['D_k'+str(i+1)]=df1_2.apply(
              lambda x: dist_e(x['v1'],x['v2'],k_v1,k_v2),
              axis=1
              )

          # asignar cluster mas cercano 
          df1_2['Cluster']=[
            j.replace('D_','') for j in 
            df1_2[['D_k'+str(x+1) for x in range(k)]].idxmin(axis=1)
            ]

          # generar objeto grafico 
          fig = go.Figure()
          lista_flechas = []

          for i in range(k):
            
            # agregar puntos regulares 
            fig.add_trace(go.Scatter(
              mode='markers',
              x=df1_2.loc[df1_2['Cluster']=='k'+str(i+1),'v1'],
              y=df1_2.loc[df1_2['Cluster']=='k'+str(i+1),'v2'],
              marker=dict(
                color=px.colors.qualitative.Plotly[i], # lista de colores 
                size=7
                ),
              showlegend=True,
              name = 'k'+str(i+1)+' ('+str(
                len(df1_2.loc[df1_2['Cluster']=='k'+str(i+1)])
                )+')'
              ))

            # agregar puntos centroide 
            fig.add_trace(go.Scatter(
              mode='markers',
              x=df2.loc[df2['Cluster']=='k'+str(i+1),'v1'],
              y=df2.loc[df2['Cluster']=='k'+str(i+1),'v2'],
              marker=dict(
                color=px.colors.qualitative.Plotly[i], # lista de colores 
                size=14,
                symbol='diamond',
                line=dict(color='black',width=1)
                ),
              showlegend=False,
              name = 'centroide-k'+str(i+1)
              ))
            
            # agregar flechas de variacion ubicacion centroide (solo aplica para iteracion>1)
            if(q>0):
              df_prev = l_centroides[q-1] # df de centroides anterior
              lista_flechas.append(
                go.layout.Annotation(dict(
                  showarrow=True,
                  ax=df_prev.loc[df_prev['Cluster']=='k'+str(i+1),'v1'].item(),
                  ay=df_prev.loc[df_prev['Cluster']=='k'+str(i+1),'v2'].item(),
                  xref='x', yref='y',text='',axref='x', ayref='y',
                  x=df2.loc[df2['Cluster']=='k'+str(i+1),'v1'].item(), # destino eje x 
                  y=df2.loc[df2['Cluster']=='k'+str(i+1),'v2'].item(), # destino eje y
                  arrowhead=3,arrowwidth=1.5,arrowcolor='black'
                ))
              )
            
          # otros ajustes 
          fig.update_layout(
            title='Iteracion '+str(q+1),
            xaxis_title='variable v1 (eje x)',
            yaxis_title='variable v2 (eje y)',
            width=400,height=400,
            legend=dict(orientation='h',yanchor='bottom',y=1,xanchor='left',x=0),
            xaxis_range=[lim_min,lim_max],
            yaxis_range=[lim_min,lim_max]
            )
          
          # agregar flechas de variacion ubicacion centroide (solo aplica para iteracion>1)
          if(q>0):
            fig.update_layout(
              annotations = lista_flechas
              )
          
          # respaldar objeto grafico 
          l_graficos.append(fig)
          
          # generar cuadro resumen de centroides 
          if(q>0): # si ya es segunda iteracion o mayor 

            df_varc = pd.merge(
              l_centroides[q], 
              l_centroides[q-1],  
              how='left', 
              on=['Cluster']
              )

            df_varc['actual']=df_varc.apply(lambda x:
              '( '+str(round(x['v1_x'],1))+' , '+str(round(x['v2_x'],1))+' )',
              axis=1
              )

            df_varc['anterior']=df_varc.apply(lambda x:
              '( '+str(round(x['v1_y'],1))+' , '+str(round(x['v2_y'],1))+' )',
              axis=1
              )

            df_varc['var_x']=df_varc.apply(lambda x:
              (x['v1_x']-x['v1_y'])/x['v1_y'],
              axis=1
              )

            df_varc['var_x']=df_varc.apply(lambda x:
              (x['v1_x']-x['v1_y'])/x['v1_y'],
              axis=1
              )

            df_varc['var_y']=df_varc.apply(lambda x:
              (x['v2_x']-x['v2_y'])/x['v2_y'],
              axis=1
              )

            df_varc['variacion%']=df_varc.apply(lambda x:
              '( '+str(round(100*x['var_x'],1))+'% , '+str(round(100*x['var_y'],1))+'% )',
              axis=1
              )

            df_varc['distancia']=df_varc.apply(lambda x:
              round(dist_e(x['v1_x'],x['v2_x'],x['v1_y'],x['v2_y']),1),
              axis=1
              )
            
            # respaldar cuadro de centroides 
            l_centroides2.append(df_varc[['Cluster','actual','anterior','variacion%','distancia']])
            
          else: # si es primera iteracion 
            
            df_varc = df2.copy()

            df_varc['actual']=df_varc.apply(lambda x:
              '( '+str(round(x['v1'],1))+' , '+str(round(x['v2'],1))+' )',
              axis=1
              )
          
            # respaldar cuadro de centroides 
            l_centroides2.append(df_varc[['Cluster','actual']])



          # recalcular valor de centroides segun valores promedio
          df2_2 = df1_2.groupby(['Cluster']).agg( 
              N = pd.NamedAgg(column = 'Cluster', aggfunc = len),
              v1 = pd.NamedAgg(column = 'v1', aggfunc = np.mean),
              v2 = pd.NamedAgg(column = 'v2', aggfunc = np.mean)
          )
          df2_2.reset_index(level=df2_2.index.names, inplace=True) # pasar indices a columnas
          df2 = df2_2[['Cluster','v1','v2']]
      
        return l_graficos,l_centroides2
          
    
    #-----------------------------------------------------------------------------------------------------------
    # [04.2] mostrar resultados 
    #-----------------------------------------------------------------------------------------------------------

      # generar lista de entregables 
      l_graficos,l_centroides = generar_entregables_kmeans(
        max_iter = 10,
        k = nk,
        df1 = df1
        )
      
      # generar columnas 
      col_4a = [None] * 10
      col_4b = [None] * 10
      
      # iterar agregando entregables
      for i in range(10):
        
        # agregar linea separadora 
        st.markdown('---')

        # crear columnas del momento 
        col_4a[i],col_4b[i] = st.columns((1,1))

        # mostrar grafico de iteracion   
        col_4a[i].plotly_chart(l_graficos[i], use_container_width=True) 
      
        # mostrar cuadro de iteracion
        col_4b[i].text('') # saltarse espacios 
        col_4b[i].text('') # saltarse espacios 
        col_4b[i].dataframe(l_centroides[i])

#**************************************************************************************************************
# [Z] Lanzar App
#**************************************************************************************************************

    
# arrojar main para lanzar App
if __name__=='__main__':
    main()
    
# Escribir en terminal: streamlit run App_kmeans_v3.py
# !streamlit run App_kmeans_v3.py


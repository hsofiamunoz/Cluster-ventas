!pip install scikit-learn

import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Función para cargar y procesar el archivo``
def cargar_datos(archivo):
    df = pd.read_excel(archivo)  # Ajusta el nombre de la hoja según tu archivo
    return df.set_index('PUNTO_VENTA')


def seleccionar_columnas_de_interes(df):
    opciones = df.columns.tolist()
    return opciones
    

# Función para aplicar PCA y devolver los resultados
def aplicar_pca(df):
    variables_explicativas = ['CONTRIBUCION', 'ROTACION', 'VENTA_POR_MES', 'MARGEN', 'VENTA_PESOS', 'VENTA_UNDS']
    X = df[variables_explicativas].dropna()
    # Estandarización de los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Aplicación de PCA
    pca = PCA()
    pca_result = pca.fit_transform(X_scaled)
    # DataFrame con resultados de PCA
    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])

    df_result = df.reset_index().merge(pca_df, right_index=True,left_index=True, how='left').rename(
    columns={'PC1': 'z1', 'PC2': 'z2', 'PC3': 'z3'}
    )

    # Cálculo del índice compuesto
    df_result['Indice'] = df_result['z1'] * pca.explained_variance_ratio_[0] + \
                        df_result['z2'] * pca.explained_variance_ratio_[1] + \
                        df_result['z3'] * pca.explained_variance_ratio_[2]
    
    df_result['Indice'] = df_result['Indice']*(-1)
    df_result = df_result[['PUNTO_VENTA', 'COD_PUNTO_VENTA', 'CONTRIBUCION', 'ROTACION', 'VENTA_POR_MES', 'MARGEN', 'VENTA_PESOS', 'VENTA_UNDS', 'Indice']]

    df_result = df_result.sort_values(by = 'Indice', ascending=False)
    df_result['ranking_pca'] = range(1, len(df_result) + 1)

    return df_result


def asignar_grupo_segun_indice(indice):
    if indice >= 1:
        return 'grupo1'
    elif ((indice > 0) & (indice < 1)):
        return 'grupo2'
    elif ((indice < 0) & (indice > -1)):
        return 'grupo3'
    else:
        return 'grupo4'

def grafica_resultados(df_result2):
    import plotly.graph_objects as go

    df_result2 = df_result2.sort_values(by='COD_PUNTO_VENTA', ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Line(
        x = df_result2['COD_PUNTO_VENTA'], 
        y = [-1]*len(df_result2['COD_PUNTO_VENTA']),
        line_color='red',
        name='Línea -1'
        ))
                

    fig.add_trace(go.Line(
        x = df_result2['COD_PUNTO_VENTA'], 
        y = [1]*len(df_result2['COD_PUNTO_VENTA']),
        line_color='green',
        name='Línea 1'
        ))

    fig.add_trace(go.Line(
        x = df_result2['COD_PUNTO_VENTA'], 
        y = [0]*len(df_result2['COD_PUNTO_VENTA']),
        line_color='gray',
        name='Línea 0'
        ))

    fig.add_trace(go.Scatter(
        x = df_result2['COD_PUNTO_VENTA'], 
        y = df_result2['Indice'],
        mode='markers',
        marker=dict(color='purple'),
        name='Indice tiendas',
        ))
    
    # st.plotly_chart(fig)
    return fig 

def grafica_pastel(df_result2):
    import plotly.express as px

    grafica_pie  = df_result2.groupby('Grupo')[['Indice']].count().reset_index()
    print(grafica_pie)

    colores = ['green', 'yellowgreen', 'orange', 'red']

    fig = px.pie(grafica_pie, values='Indice', names='Grupo', title='Gráfico de Pie', color='Grupo', 
                 color_discrete_sequence=colores)


    return fig

def main():
    # st.markdown("<h1 style='text-align: center;'>Análisis de Tiendas con PCA</h1>", unsafe_allow_html=True)

    # Página para cargar el archivo
    st.header('Cargar Archivo Excel')
    archivo = st.file_uploader('Cargar archivo Excel', type=['xls', 'xlsx'])

    if archivo is not None:

        df = cargar_datos(archivo)
        st.write('Muestra del archivo cargado:')
        st.write(df.head(5))

        st.header('')
        st.subheader('Seleccionar Columnas de Interés')
        opciones = seleccionar_columnas_de_interes(df)
        seleccion = st.multiselect('Selecciona opciones:', opciones)

        # Mostrar las opciones seleccionadas
        if seleccion:
            st.write(f'Has seleccionado estas columnas: {", ".join(seleccion)}')
            
        else:
            st.write('Aún no has seleccionado ninguna columna.')

        # # Página para mostrar resultados después de aplicar PCA
        st.header('')
        if st.button('Generar Resultados'):

            columnas_seleccionadas = ','.join(seleccion).split(',')
            print(columnas_seleccionadas)
            
            df_pca = aplicar_pca(df)

            df_pca['Grupo'] = df_pca['Indice'].apply(lambda x: asignar_grupo_segun_indice(x))
            df_pca.to_excel('resultados_pca.xlsx', index=False)
            
            st.write('Muestra de los resultados:')
            st.write(df_pca.head(5))

            st.write('Grafica de resultados')
            fig = grafica_resultados(df_pca)
            st.plotly_chart(fig)

            fig2 = grafica_pastel(df_pca)
            st.plotly_chart(fig2)

            if st.button('Exportar resultados'):
                df_pca.to_excel('resultados_pca.xlsx', index=False)
                st.write('Resultados exportados a Excel')

if __name__ == '__main__':
    main()

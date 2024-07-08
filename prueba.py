import streamlit as st
import ipywidgets as widgets
from IPython.display import display
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


def main():
    # st.markdown("<h1 style='text-align: center;'>Análisis de Tiendas con PCA</h1>", unsafe_allow_html=True)

    # Página para cargar el archivo
    st.header('Cargar Archivo Excel')


    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    def click_button():
        st.session_state.clicked = True

    st.button('Click me', on_click=click_button)

    if st.session_state.clicked:
        # The message and nested widget will remain on the page
        st.write('Button clicked!')
        st.slider('Select a value')

if __name__ == '__main__':
    main()

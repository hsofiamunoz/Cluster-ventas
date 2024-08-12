import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import openpyxl
from openpyxl.drawing.image import Image
from io import BytesIO
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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
    img_bytes = fig.to_image(format='png')
    return img_bytes

def to_excel(df: pd.DataFrame):
    from io import BytesIO
    in_memory_fp = BytesIO()
    df.to_excel(in_memory_fp)
    # Write the file out to disk to demonstrate that it worked.
    in_memory_fp.seek(0, 0)
    return in_memory_fp.read()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO
import zipfile
import io
import matplotlib.pyplot as plt

def create_matplotlib_plot(df_result2):
    variable_x = 'Columna1'

    plt.figure(figsize=(10, 5))

    # Add lines at y=-1, y=0, y=1
    plt.plot(df_result2[variable_x], [-1]*len(df_result2[variable_x]), color='red', label='Línea -1')
    plt.plot(df_result2[variable_x], [1]*len(df_result2[variable_x]), color='green', label='Línea 1')
    plt.plot(df_result2[variable_x], [0]*len(df_result2[variable_x]), color='gray', label='Línea 0')

    # Scatter plot for 'Indice'
    plt.scatter(df_result2[variable_x], df_result2['Columna2'], color='purple', label='Indice tiendas')

    plt.xlabel(variable_x)
    plt.ylabel('Values')
    plt.legend()
    plt.title('Graph using Matplotlib')

    # Save to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return buffer

def main():
    st.title("Generador de Archivo ZIP con Excel y Gráfico")

    # DataFrame de ejemplo
    df = pd.DataFrame({
        'Columna1': [10, 20, 30],
        'Columna2': [1, 2, 3]
    })
    st.write(df)

    fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
    st.plotly_chart(fig)

    import plotly.io as pio
    import matplotlib.pyplot as plt

    # Create a buffer to save the Matplotlib figure
    matplotlib_buffer = create_matplotlib_plot(df)
    st.download_button(
    label="Download Matplotlib Figure",
    data=matplotlib_buffer,
    file_name="matplotlib_figure.png",
    mime="image/png"
    )
    
    # Funciona ok    
    # plt.figure()
    # plt.bar([0, 1, 2], [2, 3, 1])
    # plt.title("Bar Plot")

    # buffer = io.BytesIO()
    # plt.savefig(buffer, format='png')
    # buffer.seek(0)
    # plt.close()
    
    # st.download_button(
    #     label="Download Figure",
    #     data=buffer,
    #     file_name="plotly_figure.png",
    #     mime="image/png"
    # )



if __name__ == '__main__':
    main()
























    # Generar gráficos
    # generar_grafico()

    # st.write('Siguiente')
    # st.button('Hola')

    # grafico_img = generar_grafico()

    # Descargar la imagen
    # st.download_button(
    #     label="Descargar Imagen del Gráfico",
    #     data=grafico_img,
    #     file_name="grafico.png",
    #     mime="image/png"
    # )

    # img_buffer = BytesIO(grafico_img)
    # print(img_buffer)
    # Descargar la imagen
    # st.download_button(
    #     label="Descargar Imagen del Gráfico",
    #     data=img_buffer.getvalue(),
    #     file_name="grafico.png",
    #     mime="image/png"
    # )
    # img = Image('blue-jay1.jpg')
    # buf = BytesIO()
    # img.save(buf, format="JPEG")
    # byte_im = buf.getvalue()


    # with open('blue-jay1.jpg', "rb") as file:
    #     btn = st.download_button(
    #     label="Download image",
    #     data=file,
    #     file_name="flower.png",
    #     mime="image/png"
    #     )
    
    # excel_data = to_excel(df)
    # file_name = "excel.xlsx"
    # st.download_button(
    #     f"Click to download {file_name}",
    #     excel_data,
    #     file_name,
    #     f"text/{file_name}",
    #     key=file_name
    # )




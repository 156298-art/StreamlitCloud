#Ejercicio 4: Dashboard Interactivo con Streamlit Cloud
#Paso 1: Configuración del Entorno Streamlit
#Crea un archivo Python llamado app_ventas_retail.py o un nuevo notebook y descargalo como app_ventas_retail.py:
# Install Streamlit
!pip install streamlit

# Importaciones necesarias
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="RetailMax - Predictor de Ventas",
    page_icon="         ",
    layout="wide",

    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

print("   Configuración inicial de Streamlit completada")

#Paso 2: Función de Carga del Modelo
@st.cache_data
def cargar_datos_demo():

    """
    Carga datos de demostración para la app
    """
    # Crear datos sintéticos para demostración
    np.random.seed(42)

    tiendas = [f"Tienda_{i:02d}" for i in range(1, 21)]
    fechas = pd.date_range(start='2024-01-01', end='2024-03-15', freq='W')

    data = []
    for tienda in tiendas:

        for fecha in fechas:

            data.append({
                'tienda_id': tienda,
                'fecha': fecha,
                'ventas_semanales': np.random.normal(15000, 3000),
                'promocion_activa': np.random.choice([0, 1], p=[0.7, 0.3]),
                'inventario_inicial': np.random.normal(50000, 10000),
                'temperatura_promedio': np.random.normal(20, 8),
                'año': fecha.year,
                'mes': fecha.month,
                'semana': fecha.isocalendar().week,
                'dia_semana': fecha.dayofweek
            })

    return pd.DataFrame(data)

@st.cache_resource
def cargar_modelo_demo():

    """
    Carga o crea un modelo de demostración
    """
    # Para demostración, creamos un modelo simple
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    # Cargar datos
    df = cargar_datos_demo()

    # Preparar features
    feature_columns = ['promocion_activa', 'inventario_inicial', 'temperatura_promedio',
                      'año', 'mes', 'semana', 'dia_semana']
    X = df[feature_columns]
    y = df['ventas_semanales']

    # Entrenar modelo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    return modelo, feature_columns

# Cargar modelo y datos
modelo, feature_columns = cargar_modelo_demo()
df_historico = cargar_datos_demo()

st.success("   Modelo y datos cargados exitosamente")

#Paso 3: Sidebar con Controles de Entrada

def crear_sidebar():

    """
    Crea la barra lateral con controles de entrada
    """
    st.sidebar.markdown("##                  Configuración de Predicción")

    # Selección de tienda
    tiendas_disponibles = df_historico['tienda_id'].unique()
    tienda_seleccionada = st.sidebar.selectbox(
        "          Selecciona la tienda:",
        options=tiendas_disponibles,
        help="Elige la tienda para la cual quieres hacer la predicción"
    )

    # Fecha de predicción
    fecha_prediccion = st.sidebar.date_input(
        "             Fecha de la semana a predecir:",
        value=datetime.now() + timedelta(days=7),
        help="Selecciona la fecha de inicio de la semana"
    )

    # Promoción activa
    promocion_activa = st.sidebar.radio(
        "       ¿Habrá promoción activa?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Sí",
        help="Indica si habrá una promoción especial durante la semana"
    )

    # Inventario inicial
    inventario_inicial = st.sidebar.slider(
        "       Inventario inicial (unidades):",
        min_value=20000,
        max_value=80000,
        value=50000,
        step=1000,
        help="Cantidad de productos disponibles al inicio de la semana"
    )

    # Temperatura promedio
    temperatura_promedio = st.sidebar.slider(
        "     Temperatura promedio esperada (°C):",
        min_value=-10,
        max_value=40,
        value=20,
        step=1,
        help="Temperatura promedio esperada durante la semana"
    )
    # Botón de predicción
    predecir = st.sidebar.button(
        "      Hacer Predicción",
        type="primary",
        help="Haz clic para generar la predicción con los parámetros seleccionados"
    )

    return {
        'tienda': tienda_seleccionada,
        'fecha': fecha_prediccion,
        'promocion': promocion_activa,
        'inventario': inventario_inicial,
        'temperatura': temperatura_promedio,
        'predecir': predecir
    }

# Crear sidebar
parametros = crear_sidebar()

#Paso 4: Función de Predicción
def hacer_prediccion(parametros):

    """
    Realiza la predicción basada en los parámetros de entrada
    """
    # Preparar datos para predicción
    fecha = pd.to_datetime(parametros['fecha'])

    datos_prediccion = pd.DataFrame({
        'promocion_activa': [parametros['promocion']],
        'inventario_inicial': [parametros['inventario']],
        'temperatura_promedio': [parametros['temperatura']],
        'año': [fecha.year],
        'mes': [fecha.month],
        'semana': [fecha.isocalendar().week],
        'dia_semana': [fecha.dayofweek]
    })

    # Hacer predicción
    prediccion = modelo.predict(datos_prediccion)[0]

    # Calcular intervalo de confianza (simulado)
    std_error = prediccion * 0.15  # 15% de error estándar estimado
    intervalo_inferior = prediccion - 1.96 * std_error
    intervalo_superior = prediccion + 1.96 * std_error

    return {
        'prediccion': prediccion,
        'intervalo_inferior': max(0, intervalo_inferior),
        'intervalo_superior': intervalo_superior,
        'confianza': 95
    }

def mostrar_resultados_prediccion(resultado, parametros):

    """
    Muestra los resultados de la predicción de manera atractiva
    """
    st.markdown("##       Resultados de la Predicción")

    # Métricas principales
    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            label="   Ventas Predichas",
            value=f"${resultado['prediccion']:,.0f}",
            help="Predicción puntual de ventas para la semana"
        )

    with col2:

        st.metric(
            label="    Rango Mínimo",
            value=f"${resultado['intervalo_inferior']:,.0f}",
            help=f"Límite inferior del intervalo de confianza al {resultado['confianza']}%"
        )

    with col3:

        st.metric(
            label="    Rango Máximo",
            value=f"${resultado['intervalo_superior']:,.0f}",
            help=f"Límite superior del intervalo de confianza al {resultado['confianza']}%"
        )

    # Gráfico de gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = resultado['prediccion'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Ventas Predichas ($)"},
        delta = {'reference': df_historico[df_historico['tienda_id'] == parametros['tienda']]['ventas_semanales'].mean()},
        gauge = {
            'axis': {'range': [None, 30000]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 10000], 'color': "lightgray"},
                {'range': [10000, 20000], 'color': "gray"},
                {'range': [20000, 30000], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': resultado['prediccion']
            }
        }
    ))

    fig_gauge.update_layout(height=400)
    st.plotly_chart(fig_gauge, use_container_width=True)

#Paso 5: Análisis Comparativo
def crear_analisis_comparativo(parametros, resultado):

    """
    Crea análisis comparativo con datos históricos
    """
    st.markdown("##     Análisis Comparativo")

    # Filtrar datos históricos de la tienda
    datos_tienda = df_historico[df_historico['tienda_id'] == parametros['tienda']].copy()
    datos_tienda = datos_tienda.sort_values('fecha')

    # Crear gráfico de serie temporal
    fig_temporal = go.Figure()

    # Ventas históricas
    fig_temporal.add_trace(go.Scatter(
        x=datos_tienda['fecha'],
        y=datos_tienda['ventas_semanales'],
        mode='lines+markers',
        name='Ventas Históricas',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))

    # Predicción
    fig_temporal.add_trace(go.Scatter(
        x=[parametros['fecha']],
        y=[resultado['prediccion']],
        mode='markers',
        name='Predicción',
        marker=dict(color='red', size=12, symbol='star')
    ))



    # Intervalo de confianza
    fig_temporal.add_trace(go.Scatter(
        x=[parametros['fecha'], parametros['fecha']],
        y=[resultado['intervalo_inferior'], resultado['intervalo_superior']],
        mode='lines',
        name=f'Intervalo {resultado["confianza"]}%',
        line=dict(color='red', dash='dash'),
        showlegend=False
    ))

    fig_temporal.update_layout(
        title=f'Evolución de Ventas - {parametros["tienda"]}',
        xaxis_title='Fecha',
        yaxis_title='Ventas Semanales ($)',
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig_temporal, use_container_width=True)

    # Estadísticas comparativas
    col1, col2 = st.columns(2)

    with col1:

        st.markdown("###       Estadísticas Históricas")
        promedio_historico = datos_tienda['ventas_semanales'].mean()
        std_historico = datos_tienda['ventas_semanales'].std()

        st.write(f"**Promedio histórico:** ${promedio_historico:,.0f}")
        st.write(f"**Desviación estándar:** ${std_historico:,.0f}")
        st.write(f"**Ventas máximas:** ${datos_tienda['ventas_semanales'].max():,.0f}")
        st.write(f"**Ventas mínimas:** ${datos_tienda['ventas_semanales'].min():,.0f}")

    with col2:

        st.markdown("###        Comparación con Predicción")
        diferencia = resultado['prediccion'] - promedio_historico
        porcentaje_diferencia = (diferencia / promedio_historico) * 100

        if diferencia > 0:

            st.success(f"**${diferencia:,.0f}** por encima del promedio ({porcentaje_diferencia:+.1f}%)")
        else:

            st.warning(f"**${abs(diferencia):,.0f}** por debajo del promedio ({porcentaje_diferencia:+.1f}%)")

        # Percentil de la predicción
        percentil = (datos_tienda['ventas_semanales'] < resultado['prediccion']).mean() * 100
        st.info(f"La predicción está en el **percentil {percentil:.0f}** de ventashistóricas")

def crear_analisis_sensibilidad():

    """
    Crea análisis de sensibilidad para diferentes escenarios
    """
    st.markdown("##     Análisis de Sensibilidad")

    st.markdown("Explora cómo diferentes factores afectan las predicciones:")

    # Análisis de promociones
    escenarios_promocion = []
    for promo in [0, 1]:

        datos_escenario = pd.DataFrame({
            'promocion_activa': [promo],
            'inventario_inicial': [50000],
            'temperatura_promedio': [20],
            'año': [2024],
            'mes': [3],
            'semana': [12],
            'dia_semana': [0]
        })
        pred = modelo.predict(datos_escenario)[0]
        escenarios_promocion.append({
            'Escenario': 'Con Promoción' if promo else 'Sin Promoción',
            'Ventas_Predichas': pred
        })

    df_promocion = pd.DataFrame(escenarios_promocion)

    fig_promocion = px.bar(
        df_promocion,
        x='Escenario',
        y='Ventas_Predichas',
        title='Impacto de las Promociones en las Ventas',
        color='Ventas_Predichas',
        color_continuous_scale='Blues'
    )

    fig_promocion.update_layout(height=400)
    st.plotly_chart(fig_promocion, use_container_width=True)

    # Mostrar diferencia
    diferencia_promocion = df_promocion.iloc[1]['Ventas_Predichas'] - df_promocion.iloc[0]['Ventas_Predichas']
    st.info(f"      **Insight:** Las promociones aumentan las ventas en aproximadamente **${diferencia_promocion:,.0f}** ({(diferencia_promocion/df_promocion.iloc[0]['Ventas_Predichas']*100):+.1f}%)")

#Paso 6: Página Principal de la App
def main():

    """
    Función principal de la aplicación
    """
    # Header principal
    st.markdown('<h1 class="main-header">          RetailMax - Predictor de Ventas</h1>',unsafe_allow_html=True)

    st.markdown("""
    ### Bienvenido al Sistema de Predicción de Ventas

    Esta herramienta te permite predecir las ventas semanales de cualquier tienda R
etailMax
    basándose en factores como promociones, inventario y condiciones climáticas.

    **Instrucciones:**

    1.           Configura los parámetros en la barra lateral
    2.       Haz clic en "Hacer Predicción"
    3.       Analiza los resultados y comparaciones
    """)

    # Verificar si se debe hacer predicción
    if parametros['predecir']:

        with st.spinner('    Generando predicción...'):

            resultado = hacer_prediccion(parametros)

            # Mostrar resultados
            mostrar_resultados_prediccion(resultado, parametros)

            # Análisis comparativo
            crear_analisis_comparativo(parametros, resultado)

            # Análisis de sensibilidad
            crear_analisis_sensibilidad()

            # Botón de descarga de resultados
            st.markdown("##        Descargar Resultados")

            resultados_descarga = {
                'Tienda': parametros['tienda'],
                'Fecha_Prediccion': str(parametros['fecha']),
                'Promocion_Activa': 'Sí' if parametros['promocion'] else 'No',
                'Inventario_Inicial': parametros['inventario'],
                'Temperatura_Promedio': parametros['temperatura'],
                'Ventas_Predichas': f"${resultado['prediccion']:,.0f}",
                'Rango_Minimo': f"${resultado['intervalo_inferior']:,.0f}",
                'Rango_Maximo': f"${resultado['intervalo_superior']:,.0f}",
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            df_descarga = pd.DataFrame([resultados_descarga])
            csv_descarga = df_descarga.to_csv(index=False)

            st.download_button(
                label="     Descargar Predicción (CSV)",
                data=csv_descarga,
                file_name=f"prediccion_ventas_{parametros['tienda']}_{parametros['fecha']}.csv",mime="text/csv"
            )

    else:

        # Mostrar información general cuando no hay predicción
        st.markdown("##     Dashboard General")

        # Métricas generales
        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric("          Total Tiendas", len(df_historico['tienda_id'].unique()))

        with col2:

            st.metric("             Semanas de Datos", len(df_historico['fecha'].unique()))

        with col3:

            promedio_general = df_historico['ventas_semanales'].mean()
            st.metric("   Promedio Ventas", f"${promedio_general:,.0f}")

        with col4:

            total_ventas = df_historico['ventas_semanales'].sum()
            st.metric("       Ventas Totales", f"${total_ventas:,.0f}")

        # Gráfico de ventas por tienda
        ventas_por_tienda = df_historico.groupby('tienda_id')['ventas_semanales'].mean().sort_values(ascending=False)

        fig_tiendas = px.bar(
            x=ventas_por_tienda.index,
            y=ventas_por_tienda.values,
            title='Promedio de Ventas por Tienda',
            labels={'x': 'Tienda', 'y': 'Ventas Promedio ($)'}
        )

        fig_tiendas.update_layout(height=500)
        st.plotly_chart(fig_tiendas, use_container_width=True)

# Ejecutar aplicación principal
if __name__ == "__main__":

    main()

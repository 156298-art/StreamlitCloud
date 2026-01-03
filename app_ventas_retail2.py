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
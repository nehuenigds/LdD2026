# Laboratorio de regresión · TP1

Hacé doble clic en **Abrir app.cmd**. La aplicación se abre en el navegador en http://127.0.0.1:8768 y funciona sin conexión a internet. No necesita mantener abierto el notebook.

## Cómo explorar

- Hay tres modelos iniciales. Elegí uno en el panel izquierdo y activá o desactivá sus variables.
- **+ Comparar** duplica el modelo seleccionado, hasta cuatro configuraciones simultáneas. Podés cambiar sus nombres.
- Agregá cuadrados, interacciones o umbrales. Los umbrales tienen un deslizador que recalcula el ajuste.
- El círculo a la izquierda del modelo controla su visibilidad en los gráficos, sin quitarlo de la comparación de MAE.
- Elegí entre **Curvas**, **Real vs. predicho** y **Residuos**. La rueda hace zoom, arrastrar desplaza el gráfico y el doble clic restablece la vista.
- El selector de eje cambia la visualización, no las variables de entrenamiento.
- El MAE global y los cinco MAE por fold se actualizan al modificar un modelo. Δ usa el primer modelo de la lista como referencia. No es una mejora porcentual.
- Las ecuaciones y R² están al pie. Ridge es opcional, con intensidad ajustable.

## Evaluación

Se mantiene el procedimiento del notebook: separación 80/20 con semilla 42 y validación cruzada de cinco folds dentro de los 540 registros de desarrollo, también con semilla 42. La reserva de 135 registros y `X_test.csv` no se evalúan. No se eliminan pesos bajos. La estandarización se ajusta únicamente sobre el entrenamiento de cada fold.

El MAE de validación se calcula con predicciones sobre casos no usados para ajustar ese modelo. Las curvas y ecuaciones, en cambio, utilizan un ajuste sobre los 540 registros y un perfil con medianas numéricas y modas categóricas. Son ilustrativas; no son predicciones de validación. Al variar `mage`, se actualiza `mature_bin` de forma coherente con el umbral de 35 años. Las otras variables se mantienen en el perfil indicado.

Sin variables se usa la mediana del entrenamiento de cada fold, una referencia trivial que no sería una entrega válida. Las selecciones de la app viven en la pestaña: recargar restablece las tres variantes iniciales. La app no modifica los datos ni los notebooks y no genera una entrega automáticamente.

Comparar muchas configuraciones puede adaptar la elección a estos folds. Una mejora pequeña aquí no garantiza una mejora en datos nuevos. Los coeficientes no demuestran efectos causales.

## Entorno

El lanzador usa el Python instalado en `C:\Python314\pythonw.exe`. En otro equipo, se puede ejecutar `python servidor.py --open`, con numpy, pandas, scikit-learn y plotly instalados. Plotly se sirve desde la instalación local; no se descargan scripts externos.

El proceso escucha solo en 127.0.0.1. Cerrar la pestaña no detiene el proceso; abrir el lanzador otra vez reutiliza la instancia. Para detenerlo, finalizá el proceso Python de `servidor.py` desde el Administrador de tareas. Si el puerto está ocupado por otra app, ejecutá `python servidor.py --port 8769 --open`.

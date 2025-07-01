# Signature Predictor

Este proyecto es un intérprete de lenguaje de señas en tiempo real que convierte gestos de manos capturados por una cámara en texto, utilizando MediaPipe para la detección de puntos clave y un modelo de TensorFlow para la predicción.

## Tecnologías Utilizadas

*   **Python:** 3.11.9
*   **MediaPipe:** Para la detección y seguimiento de puntos clave de la mano.
*   **TensorFlow/Keras:** Para el entrenamiento y la inferencia del modelo de reconocimiento de gestos.
*   **Scikit-learn:** Para el preprocesamiento de datos (escalado y codificación de etiquetas).
*   **OpenCV (cv2):** Para la captura de video y la visualización en tiempo real.
*   **Numpy y Pandas:** Para la manipulación y análisis de datos.
*   **Joblib:** Para la serialización de los objetos `LabelEncoder` y `StandardScaler`.

## Descripción del Proyecto

Este proyecto se centra en la predicción de firmas o gestos únicos, transformando movimientos de manos capturados en tiempo real en una representación digital. Utiliza MediaPipe para la detección precisa de puntos clave de la mano y un modelo de aprendizaje profundo basado en TensorFlow para analizar y predecir el gesto o firma. La aplicación está diseñada para capturar el flujo de video, procesar las coordenadas de los puntos clave y mostrar la predicción en pantalla, ofreciendo una solución para la autenticación o identificación basada en gestos. El modelo ha sido entrenado con un dataset de puntos clave de manos, y se ha implementado un aumento de datos (horizontal flip) para mejorar la robustez del modelo ante diferentes orientaciones de manos (izquierda/derecha).

## Estructura del Proyecto

```
.
├── app/
│   ├── main.py                 # Aplicación principal para la interpretación en tiempo real
│   └── utils/
│       └── prediction.py       # Clases para la transformación de datos y predicción del modelo
├── model/
│   ├── data/                   # Datos de entrenamiento (raw y processed)
│   ├── notebooks/              # Notebooks de Jupyter para el entrenamiento del modelo
│   │   └── signclusive-mediapipe-model.ipynb
│   └── saved_models/           # Modelos entrenados, scaler y label encoder
├── .gitignore                  # Archivo para ignorar archivos y directorios en Git
├── requirements.txt            # Dependencias de Python del proyecto
└── Makefile                    # Archivo para automatizar tareas (ej. instalación de dependencias)
```

## Instalación y Ejecución

1.  **Clonar el repositorio:**
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd final
    ```
2.  **Instalar dependencias:**
    Asegúrate de tener `make` instalado.
    ```bash
    make install-deps
    ```
    (Alternativamente, puedes usar `pip install -r requirements.txt`)

3.  **Entrenar el modelo (opcional, si quieres re-entrenar o usar tu propio dataset):**
    Abre y ejecuta el notebook `model/notebooks/signclusive-mediapipe-model.ipynb` para entrenar el modelo y generar los archivos necesarios en `app/trained_model/`.

4.  **Ejecutar la aplicación:**
    ```bash
    make run
    ```

## Pruebas y Ejemplos

Aquí puedes agregar capturas de pantalla o GIFs de la aplicación en funcionamiento, mostrando diferentes gestos y las predicciones.

![Ejemplo de Predicción 1](path/to/your/image1.png)
*Descripción de la imagen 1: Predicción del gesto 'A'.*

![Ejemplo de Predicción 2](path/to/your/image2.gif)
*Descripción de la imagen 2: Demostración de varios gestos y sus predicciones.*

---

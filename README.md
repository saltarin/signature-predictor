# Signature Predictor

Este proyecto es un intérprete de lenguaje de señas en tiempo real que convierte gestos de manos capturados por una cámara en texto, utilizando MediaPipe para la detección de puntos clave y un modelo de TensorFlow para la predicción. Generado con gemini-cli.

## Tecnologías Utilizadas

*   **Python:** 3.11.9
*   **MediaPipe:** Para la detección y seguimiento de puntos clave de la mano.
*   **TensorFlow/Keras:** Para el entrenamiento y la inferencia del modelo de reconocimiento de gestos.
*   **Scikit-learn:** Para el preprocesamiento de datos (escalado y codificación de etiquetas).
*   **OpenCV (cv2):** Para la captura de video y la visualización en tiempo real.
*   **Numpy y Pandas:** Para la manipulación y análisis de datos.
*   **Joblib:** Para la serialización de los objetos `LabelEncoder` y `StandardScaler`.

## Descripción del Proyecto

Este proyecto se centra en la predicción de firmas o gestos únicos, transformando movimientos de manos capturados en tiempo real en una representación digital. Utiliza MediaPipe para la detección precisa de puntos clave de la mano y un modelo de aprendizaje profundo basado en TensorFlow para analizar y predecir el gesto o firma. La aplicación está diseñada para capturar el flujo de video, procesar las coordenadas de los puntos clave y mostrar la predicción en pantalla, ofreciendo una solución para la autenticación o identificación basada en gestos. El modelo ha sido entrenado con un dataset de puntos clave de manos, y se ha implementado un aumento de datos (horizontal flip) para mejorar la robustez del modelo ante diferentes orientaciones de manos (izquierda/derecha). El modelo fue entrenado utilizando el dataset [Signclusive MediaPipe](https://www.kaggle.com/datasets/ahmedkhairy11/signclusive-mediapipe) de Kaggle.

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

![Ejemplo de Predicción 1](https://private-user-images.githubusercontent.com/13721664/460842570-920c6466-287c-44e8-8711-1517fb866642.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTEzNDA0OTUsIm5iZiI6MTc1MTM0MDE5NSwicGF0aCI6Ii8xMzcyMTY2NC80NjA4NDI1NzAtOTIwYzY0NjYtMjg3Yy00NGU4LTg3MTEtMTUxN2ZiODY2NjQyLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA3MDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNzAxVDAzMjMxNVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWNiNTA4NzY1NTM5MWYwYzE5ZmUwNDhlYWExYmM3ODA4OWZhNTkwNTEwMjJmM2FmY2Y2MmVhZGMwMDNlNWE2YjYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.xfyCQTMgm8qyJwVPPKZOgQOOItZ2ua69z4uJIi2hSjg)
*Descripción de la imagen 1: Predicción del gesto 'C'.*

![Ejemplo de Predicción 2](https://private-user-images.githubusercontent.com/13721664/460842712-8443d3bf-dbe2-4bc9-ac20-f35bfd6b40d1.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTEzNDA0OTUsIm5iZiI6MTc1MTM0MDE5NSwicGF0aCI6Ii8xMzcyMTY2NC80NjA4NDI3MTItODQ0M2QzYmYtZGJlMi00YmM5LWFjMjAtZjM1YmZkNmI0MGQxLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA3MDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNzAxVDAzMjMxNVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWU2OTQxNjg3ZTMyY2Q4ZjcwMjhkYzUyMzU5YzVmODc2ZTM3YmI2YTc2ZjA2YmVhZmE2OGE0MGM5MTNmMTUxOTkmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.59I5fFF_cYgj0adULHGxzCU7zM1w9tvYinHB7jeqMdY)
*Descripción de la imagen 2: Predicción del gesto 'H'.*

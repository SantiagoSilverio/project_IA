Análisis de Emociones en Texto
Este proyecto utiliza un modelo de aprendizaje automático para clasificar texto en diferentes emociones. El modelo está entrenado para identificar siete emociones distintas basadas en el texto ingresado por el usuario.

Índice
Descripción del Proyecto
Instalación
Uso
Estructura de Archivos
Contribuciones
Licencia
Descripción del Proyecto
Este proyecto entrena un modelo de red neuronal para clasificar textos en diferentes categorías emocionales. El modelo se guarda y se puede cargar para realizar predicciones en tiempo real. Los usuarios pueden proporcionar retroalimentación para mejorar el modelo, que se guarda en un archivo CSV para futuras referencias.

Instalación
Clona el Repositorio

bash
Copiar código
git clone https://github.com/tu_usuario/tu_repositorio.git
cd tu_repositorio
Instala las Dependencias

Este proyecto requiere tensorflow, numpy, pandas, matplotlib, y otras bibliotecas para funcionar. Puedes instalar las dependencias utilizando pip.

bash
Copiar código
pip install tensorflow numpy pandas matplotlib
Preparar Datos

Asegúrate de tener el archivo emociones_grande.csv en la ruta correcta dentro de Google Drive. Este archivo debe contener dos columnas: texto y etiqueta.

Uso
Entrenamiento del Modelo

Si el modelo no existe en el archivo modelo_emociones.h5, se creará y entrenará automáticamente. De lo contrario, se cargará el modelo existente para hacer predicciones.

python
Copiar código
from google.colab import drive
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

drive.mount('/content/drive')

csv_path = '/content/drive/My Drive/emociones_grande.csv'
modelo_path = '/content/drive/My Drive/modelo_emociones.h5'
feedback_csv_path = '/content/drive/My Drive/feedback_emociones.csv'

data = pd.read_csv(csv_path)
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['texto'])
sequences = tokenizer.texts_to_sequences(data['texto'])
padded_sequences = pad_sequences(sequences, padding='post')
etiquetas = np.array(data['etiqueta'])

def guardar_modelo(modelo):
    modelo.save(modelo_path)
    print(f"Modelo guardado en {modelo_path}")

def cargar_modelo():
    try:
        modelo = load_model(modelo_path)
        print(f"Modelo cargado desde {modelo_path}")
        return modelo
    except IOError:
        print(f"No se encontró el modelo en {modelo_path}. Se creará un nuevo modelo.")
        return None

modelo = cargar_modelo()

if modelo is None:
    modelo = Sequential([
        Embedding(input_dim=10000, output_dim=64, input_length=padded_sequences.shape[1]),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(256, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(128)),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(7, activation='softmax')
    ])
    modelo.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    historial = modelo.fit(padded_sequences, etiquetas, epochs=10, verbose=1)
    guardar_modelo(modelo)

plt.xlabel("# Época")
plt.ylabel("Pérdida")
plt.plot(historial.history["loss"])
plt.show()

emociones = ['tristeza', 'gracioso', 'romántico', 'feliz', 'miedo', 'ira', 'vergüenza']
frases_incorrectas = []
emociones_correctas = []

while True:
    nueva_frase = [input("Ingresa una frase para analizar la emoción (o escribe 'salir' para terminar): ")]

    if nueva_frase[0].lower() == 'salir':
        print("Finalizando el análisis.")
        break

    nueva_secuencia = tokenizer.texts_to_sequences(nueva_frase)
    nueva_secuencia_padded = pad_sequences(nueva_secuencia, maxlen=padded_sequences.shape[1], padding='post')

    prediccion = modelo.predict(nueva_secuencia_padded)
    etiqueta_predicha = np.argmax(prediccion)

    print(f"Emoción predicha: {emociones[etiqueta_predicha]}")

    feedback = input("¿Es esta predicción correcta? (sí/no): ")

    if feedback.lower() == "no":
        emocion_correcta = input("¿Cuál es la emoción correcta? ")
        if emocion_correcta in emociones:
            frases_incorrectas.append(nueva_frase[0])
            emociones_correctas.append(emociones.index(emocion_correcta))
            print("Retroalimentación recibida y almacenada.")
        else:
            print("Emoción correcta no reconocida.")

    else:
        print("Predicción correcta según la retroalimentación.")

    if frases_incorrectas:
        nuevas_secuencias = tokenizer.texts_to_sequences(frases_incorrectas)
        nuevas_secuencias_padded = pad_sequences(nuevas_secuencias, maxlen=padded_sequences.shape[1], padding='post')
        nuevas_etiquetas = np.array(emociones_correctas)
        combinado_secuencias = np.concatenate([padded_sequences, nuevas_secuencias_padded])
        combinado_etiquetas = np.concatenate([etiquetas, nuevas_etiquetas])
        historial = modelo.fit(combinado_secuencias, combinado_etiquetas, epochs=5, verbose=1)
        guardar_modelo(modelo)
        feedback_data = pd.DataFrame({
            'texto': frases_incorrectas,
            'etiqueta': emociones_correctas
        })
        feedback_data.to_csv(feedback_csv_path, index=False)
        print(f"Datos de feedback guardados en '{feedback_csv_path}'.")
        frases_incorrectas.clear()
        emociones_correctas.clear()

print("Entrenamiento incrementado finalizado.")
Realizar Predicciones

Ejecuta el código para realizar predicciones sobre nuevas frases y proporciona retroalimentación para mejorar el modelo.

Estructura de Archivos
emociones_grande.csv: Archivo CSV con datos de entrenamiento.
modelo_emociones.h5: Archivo para guardar el modelo entrenado.
feedback_emociones.csv: Archivo para guardar datos de retroalimentación.
notebooks/: Carpeta opcional para notebooks de Jupyter o Google Colab.
Contribuciones
Si deseas contribuir al proyecto, por favor, envía un pull request o abre un issue para discutir los cambios que quieres proponer.

Licencia
Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

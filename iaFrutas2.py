import tensorflow as tf
import numpy as np
from PIL import Image
import os
import shutil
from sklearn.model_selection import train_test_split

# Configuración inicial
width, height = 150, 150  # Tamaño de las imágenes
ruta_datos = r'C:\Users\Natalia\Desktop\Universidad\2024-2\Inteligencia Artificial\proyectoIA\fotos IA'  # Ruta principal de las carpetas por clase
clases = ["apple", "banana", "mango", "orange", "peach", "pear", "tomatoes"]
num_clases = len(clases)

# Rutas para conjunto de entrenamiento y prueba
ruta_train = os.path.join(ruta_datos, "train")
ruta_test = os.path.join(ruta_datos, "test")

# Crear carpetas de entrenamiento y prueba
os.makedirs(ruta_train, exist_ok=True)
os.makedirs(ruta_test, exist_ok=True)

for clase in clases:
    os.makedirs(os.path.join(ruta_train, clase), exist_ok=True)
    os.makedirs(os.path.join(ruta_test, clase), exist_ok=True)

# Dividir imágenes en entrenamiento y prueba
for clase in clases:
    ruta_clase = os.path.join(ruta_datos, clase)
    imagenes = [os.path.join(ruta_clase, archivo) for archivo in os.listdir(ruta_clase) if os.path.isfile(os.path.join(ruta_clase, archivo))]
    
    # División en 70% entrenamiento y 30% prueba
    train_files, test_files = train_test_split(imagenes, test_size=0.3, random_state=42)
    
    # Copiar imágenes a sus respectivas carpetas
    for archivo in train_files:
        shutil.copy(archivo, os.path.join(ruta_train, clase))
    for archivo in test_files:
        shutil.copy(archivo, os.path.join(ruta_test, clase))

print("Imágenes divididas en entrenamiento y prueba.")

# Carga y preprocesamiento de datos
def cargar_datos(ruta_base, clases, width, height):
    imagenes = []
    etiquetas = []
    for i, clase in enumerate(clases):
        ruta_clase = os.path.join(ruta_base, clase)
        for archivo in os.listdir(ruta_clase):
            ruta_imagen = os.path.join(ruta_clase, archivo)
            try:
                # Carga y redimensiona la imagen
                img = Image.open(ruta_imagen).convert("RGB")
                img = img.resize((width, height))
                img_array = np.array(img) / 255.0  # Normalización
                imagenes.append(img_array)
                etiquetas.append(i)
            except Exception as e:
                print(f"Error cargando la imagen {ruta_imagen}: {e}")
    return np.array(imagenes), tf.keras.utils.to_categorical(etiquetas, num_clases)

# Cargar datos de entrenamiento y prueba
x_train, y_train = cargar_datos(ruta_train, clases, width, height)
x_test, y_test = cargar_datos(ruta_test, clases, width, height)

# Definición del modelo
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_clases, activation='softmax')  # Salida para 7 clases
])

# Compilación del modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluación
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Pérdida: {loss}, Precisión: {accuracy}")

# Guardar el modelo
model.save('modelo_frutas.keras')

# Predicción de una nueva imagen
def predecir_imagen(ruta_imagen):
    img = Image.open(ruta_imagen).convert("RGB")
    img = img.resize((width, height))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Agregar dimensión para batch
    prediccion = model.predict(img_array)[0]
    clase = clases[np.argmax(prediccion)]
    confianza = max(prediccion) * 100
    return clase, confianza

# Prueba con una imagen nueva (ajustado para usar imágenes desde cualquier ruta)
nueva_imagen = r'C:\Users\Natalia\Desktop\Universidad\2024-2\Inteligencia Artificial\proyectoIA\fotos IA\apple\Apple1.jpg'
clase, confianza = predecir_imagen(nueva_imagen)
print(f"Clase predicha: {clase}, Confianza: {confianza:.2f}%")

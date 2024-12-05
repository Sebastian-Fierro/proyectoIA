import os
import shutil
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Configuración inicial
width, height = 150, 150  # Tamaño de las imágenes
# Ruta principal de las carpetas por clase
ruta_datos = r'C:\Users\USUARIO\Documents\Workspace\ProyectoIA\proyectoIA\fotos IA'
clases = ["apple", "banana", "orange", "peach", "pear", "tomatoes"]
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
    imagenes = [os.path.join(ruta_clase, archivo) for archivo in os.listdir(
        ruta_clase) if os.path.isfile(os.path.join(ruta_clase, archivo))]

    # División en 70% entrenamiento y 30% prueba
    train_files, test_files = train_test_split(
        imagenes, test_size=0.3, random_state=42)

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
    return np.array(imagenes), np.array(etiquetas)


# Cargar datos de entrenamiento y prueba
x_train, y_train = cargar_datos(ruta_train, clases, width, height)
x_test, y_test = cargar_datos(ruta_test, clases, width, height)

# Dividir el conjunto de entrenamiento en entrenamiento y validación
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1,
                                                  random_state=42)
# Imprimir el número de imágenes en cada conjunto
print(f"Número de imágenes de entrenamiento: {len(x_train)}")
print(f"Número de imágenes de validación: {len(x_val)}")
print(f"Número de imágenes de prueba: {len(x_test)}")

# Construir la red neuronal
model = models.Sequential([
    layers.Flatten(input_shape=(width, height, 3)),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_clases, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar la red neuronal y guardar el historial
history = model.fit(x_train, y_train, epochs=15, validation_data=(x_val, y_val))

# Evaluar el rendimiento en el conjunto de prueba
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\nAccuracy en el conjunto de prueba: {test_acc}')

# Graficar la curva de aprendizaje
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Precisión en entrenamiento', color='red', marker='o')
plt.plot(history.history['val_accuracy'], label='Precisión en validación', color='green', marker='o')
plt.title("Curva de Aprendizaje")
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.legend(loc="best")
plt.grid()
plt.show()

# Obtener las predicciones en el conjunto de prueba
y_pred = np.argmax(model.predict(x_test), axis=1)

# Generar la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Mostrar la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Matriz de Confusión")
plt.show()

# Guardar el modelo
model.save('modelo_frutas.keras')

# Predicción de una nueva imagen


def predecir_imagen(ruta_imagen, width, height, model):
    img = Image.open(ruta_imagen).convert("RGB")
    img = img.resize((width, height))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Agregar dimensión para batch
    prediccion = model.predict(img_array)[0]
    clase = clases[np.argmax(prediccion)]
    confianza = max(prediccion) * 100
    return clase, confianza


# Prueba con una imagen nueva (ajustado para usar imágenes desde cualquier ruta)
nueva_imagen = r'C:\Users\USUARIO\Documents\Workspace\ProyectoIA\proyectoIA\fotos IA\test\orange\orange (12).png'
# Mostrar la imagen de prueba

clase, confianza = predecir_imagen(nueva_imagen, width, height, model)
print(f"Clase predicha: {clase}, Confianza: {confianza:.2f}%")
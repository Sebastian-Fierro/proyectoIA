import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import pathlib

# Descripción de clases y su identificador
descripcion = ("apple", "banana", "mango", "orange", "peach", "pear", "tomatoes")
clases = {"apple" : 0, "banana" : 1, "mango" : 2, "orange" : 3, "peach" : 4, "pear" : 5, "tomatoes" : 6}

# Número de imágenes de cada clase
num_img_apple = 600
num_img_banana = 600
num_img_mango = 600
num_img_orange = 600
num_img_peach = 600
num_img_pear = 600
num_img_tomatoes = 600

# 70% de las imágenes de una clase para entrenamiento
num_entrena_apple = round(num_img_apple * 0.70)
num_entrena_banana = round(num_img_banana * 0.70)
num_entrena_mango = round(num_img_mango * 0.70)
num_entrena_orange = round(num_img_orange * 0.70)
num_entrena_peach = round(num_img_peach * 0.70)
num_entrena_pear = round(num_img_pear * 0.70)
num_entrena_tomatoes = round(num_img_tomatoes * 0.70)

# 30% de las imágenes de una clase para prueba
num_prueba_apple = round(num_img_apple * 0.30)
num_prueba_banana = round(num_img_banana * 0.30)
num_prueba_mango = round(num_img_apple * 0.30)
num_prueba_orange = round(num_img_apple * 0.30)
num_prueba_peach = round(num_img_peach * 0.30)
num_prueba_pear = round(num_img_pear * 0.30)
num_prueba_tomatoes = round(num_img_tomatoes * 0.30)


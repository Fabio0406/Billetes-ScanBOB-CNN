# Clasificador de billetes bolivianos con CNN

Red neuronal convolucional (CNN) en **TensorFlow / Keras** que reconoce el valor de un **billete boliviano** a partir de una foto: 10, 20, 50, 100 o 200 bolivianos.

> Proyecto académico — Tópicos, UAGRM (2024).

## Cómo funciona

1. `data_loader.py` carga las imágenes de `data/entrenamiento` y `data/validacion`, las redimensiona a 150×150 y aplica aumento de datos.
2. `model.py` define la red:
   - Conv2D (32 filtros, 3×3) + MaxPooling
   - Conv2D (64 filtros, 2×2) + MaxPooling
   - Flatten → Dense(256, ReLU) → Dropout(0.5) → Dense(5, softmax)
3. `main.py` entrena el modelo (20 épocas, Adam, tasa de aprendizaje 0.0004) y lo guarda en `modelo/`.
4. `predecir.py` carga el modelo y muestra la clase predicha con su porcentaje de certeza.

## Datos

```
data/
├── entrenamiento/{10,20,50,100,200}/
└── validacion/{10,20,50,100,200}/
```

Cada carpeta es una clase. El conjunto actual es pequeño (4 a 8 imágenes por clase); agregar más fotos mejora bastante la precisión.

## Requisitos

- Python 3.9 a 3.12 (TensorFlow todavía no es compatible con versiones más nuevas)
- TensorFlow 2.x

## Instalación

```bash
pip install tensorflow numpy pillow
```

## Uso

Entrenar:

```bash
python main.py
```

Predecir (cambia la ruta de la imagen al final de `predecir.py`):

```bash
python predecir.py
```

Salida de ejemplo:

```
Predicción: 100 (97.35% certeza)
```

## Configuración

Los hiperparámetros (épocas, tamaño de imagen, batch, filtros y tasa de aprendizaje) están en `config.py`.

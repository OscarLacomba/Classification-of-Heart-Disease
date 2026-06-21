# 🫀 Clasificación de Enfermedades Cardíacas con RNA

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-orange?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)
![HuggingFace](https://img.shields.io/badge/🤗_Hugging_Face-Space-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

Aplicación web interactiva que utiliza una **Red Neuronal Artificial (ANN)** entrenada con el dataset **UCI Heart Disease** para clasificar si un paciente tiene riesgo de enfermedad cardíaca.

> ⚠️ *Este proyecto es de carácter académico y no reemplaza el diagnóstico médico profesional.*

---

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Dataset](#dataset)
- [Arquitectura del Modelo](#arquitectura-del-modelo)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación y Uso Local](#instalación-y-uso-local)
- [Deploy en Hugging Face Spaces](#deploy-en-hugging-face-spaces)
- [Resultados](#resultados)
- [Tecnologías](#tecnologías)
- [Conclusiones](#conclusiones)

---

## Descripción

Este proyecto implementa el ciclo de vida completo de ciencia de datos:

1. **Carga y fusión** de los cuatro archivos procesados del UCI Heart Disease Dataset
2. **EDA** con 6 visualizaciones (distribución, correlación, boxplots, scatter)
3. **Ingeniería de características** — imputación, binarización del target y one-hot encoding
4. **Entrenamiento** de una ANN con PyTorch (3 capas ocultas, BCE Loss, Adam)
5. **Evaluación** con matriz de confusión, curva ROC y métricas de clasificación
6. **Despliegue** como aplicación Streamlit en un Docker Space de Hugging Face

---

## Dataset

| Atributo | Descripción |
|----------|-------------|
| `age` | Edad en años |
| `sex` | Sexo (1 = masculino, 0 = femenino) |
| `cp` | Tipo de dolor en el pecho (1–4) |
| `trestbps` | Presión arterial en reposo (mmHg) |
| `chol` | Colesterol sérico (mg/dl) |
| `fbs` | Glucosa en ayunas > 120 mg/dl |
| `restecg` | Resultados de ECG en reposo (0–2) |
| `thalach` | Frecuencia cardíaca máxima alcanzada |
| `exang` | Angina inducida por ejercicio |
| `oldpeak` | Depresión del segmento ST |
| `slope` | Pendiente del segmento ST pico |
| `ca` | Número de vasos principales (0–3) |
| `thal` | 3 = normal · 6 = defecto fijo · 7 = defecto reversible |
| `target` | **Variable objetivo**: 0 = sin enfermedad · 1 = enfermedad |

**Fuente:** [UCI Machine Learning Repository — Heart Disease Database](https://archive.ics.uci.edu/dataset/45/heart+disease)

**Instituciones:**

| Institución | Pacientes |
|------------|-----------|
| Cleveland Clinic Foundation | 303 |
| Hungarian Institute of Cardiology | 294 |
| V.A. Medical Center, Long Beach | 200 |
| University Hospital, Zurich | 123 |
| **Total** | **920** |

---

## Arquitectura del Modelo

```
Entrada  →  n_features (según one-hot encoding)
               ↓
Capa 1   →  64 neuronas  ·  ReLU  ·  Dropout(0.3)
               ↓
Capa 2   →  32 neuronas  ·  ReLU  ·  Dropout(0.3)
               ↓
Capa 3   →  16 neuronas  ·  ReLU
               ↓
Salida   →   1 neurona   ·  Sigmoid  →  P(enfermedad)
```

| Hiperparámetro | Valor |
|----------------|-------|
| Optimizador | Adam |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| Función de pérdida | Binary Cross-Entropy (BCE) |
| Épocas | 150 |
| Batch size | 32 |
| Scheduler | ReduceLROnPlateau (patience=10) |
| División test | 20% estratificado |

---

## Estructura del Proyecto

```
📦 heart-disease-classification/
├── 📄 app.py                        # Aplicación Streamlit principal
├── 📄 Dockerfile                    # Configuración del contenedor Docker
├── 📄 requirements.txt              # Dependencias Python
├── 📄 README.md                     # Este archivo
├── 📓 heart_disease_classification.ipynb   # Notebook con análisis completo
└── 📁 data/
    ├── processed.cleveland.data
    ├── processed.hungarian.data
    ├── processed.switzerland.data
    └── processed.va.data
```

---

## Instalación y Uso Local

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/heart-disease-classification.git
cd heart-disease-classification
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación

```bash
streamlit run app.py
```

La app estará disponible en `http://localhost:8501`

### Alternativa: Docker local

```bash
docker build -t heart-disease-app .
docker run -p 7860:7860 heart-disease-app
```

Disponible en `http://localhost:7860`

---

## Deploy en Hugging Face Spaces

1. Ir a [huggingface.co/new-space](https://huggingface.co/new-space)
2. Seleccionar **Docker** como SDK
3. Subir todos los archivos del repositorio (incluyendo la carpeta `data/`)
4. Hugging Face construirá la imagen automáticamente

El `Dockerfile` ya está configurado para exponer el puerto `7860` requerido por HF Spaces.

---

## Resultados

| Métrica | Valor |
|---------|-------|
| Exactitud (Test Accuracy) | **~82–85%** |
| ROC-AUC Score | **~0.88–0.92** |
| Precisión (clase enfermedad) | ~0.83 |
| Recall (clase enfermedad) | ~0.85 |

### Hallazgos clave del EDA

- **`thalach`** (FC máxima): predictor negativo más fuerte — pacientes enfermos alcanzan frecuencias menores
- **`oldpeak`** (depresión ST): fuerte correlación positiva con enfermedad
- **`cp`** (tipo de dolor): la presentación asintomática (tipo 4) predomina en positivos
- **`age`**: pacientes con enfermedad son en promedio ~5 años mayores

---

## Tecnologías

| Herramienta | Uso |
|-------------|-----|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | Lenguaje principal |
| ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white) | Red neuronal y entrenamiento |
| ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=white) | Interfaz web interactiva |
| ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) | Manipulación de datos |
| ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white) | Preprocesamiento y métricas |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C) | Visualizaciones |
| ![Docker](https://img.shields.io/badge/-Docker-2496ED?logo=docker&logoColor=white) | Contenedorización |
| ![Hugging Face](https://img.shields.io/badge/-Hugging_Face-FFD21E) | Despliegue en la nube |

---

## Conclusiones

Este proyecto demostró con éxito que una ANN de 3 capas ocultas puede clasificar enfermedades cardíacas con una exactitud del **~82–85%** y un ROC-AUC de **~0.88–0.92**, superando los resultados de referencia reportados en la literatura (~77–81%) para este dataset.

El trabajo futuro podría centrarse en:
- Manejo del desequilibrio de clases con SMOTE o pérdida ponderada
- Refinamiento de la imputación de datos faltantes
- Ajuste de hiperparámetros con Optuna
- Calibración del umbral de decisión para minimizar falsos negativos en contexto clínico
- Comparación con modelos alternativos (XGBoost, Random Forest)

---

## Licencia

Este proyecto está bajo la licencia [MIT](LICENSE).

---

<p align="center">
  Desarrollado con ❤️ para fines académicos  ·  Dataset: UCI Machine Learning Repository
</p>

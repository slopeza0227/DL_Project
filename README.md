# Airbnb NYC Price Prediction

Proyecto de Deep Learning para predecir el precio de alojamientos en Nueva York usando datos de Airbnb. Combina feature engineering geográfico, embeddings semánticos de amenidades con modelos de lenguaje pre-entrenados y redes neuronales.

## Tabla de contenidos

- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Modelos](#modelos)
- [Resultados](#resultados)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Utilidades](#utilidades)
- [Instalación](#instalación)
- [Uso](#uso)
- [Archivos generados](#archivos-generados)

---

## Dataset

**Fuente:** [Inside Airbnb](https://insideairbnb.com/get-the-data/) — `listings.csv` de Nueva York

| Archivo | Tamaño | Descripción |
|---------|--------|-------------|
| `listings.csv` | 73 MB | Dataset original de Inside Airbnb |
| `airbnb_ny_raw.csv` | 9.1 MB | Datos con limpieza inicial |
| `Airbnb_NY_cleaned.csv` | 1.8 MB | Dataset final para modelado |

**Dimensiones:** 48,895 propiedades · Variable objetivo: `price` (USD/noche)

**Split:** 80% entrenamiento / 20% prueba (`random_state=1234`)

**Features principales:** tipo de habitación, número de camas/baños, puntajes de reseñas, ubicación geográfica, amenidades, estado de licencia, distancia a atracciones turísticas.

---

## Pipeline

```
listings.csv (73 MB)
        │
        ▼
  [1. Limpieza]
  • Eliminación de duplicados
  • Conversión de tipos (precio, fechas, booleanos)
  • Imputación de valores nulos (camas, baños, amenidades)
  • Detección de outliers (IQR + LocalOutlierFactor)
        │
        ▼
  [2. Feature Engineering]
  ┌─────────────────────────────────────────────────┐
  │  Amenidades (NLP + Clustering)                  │
  │  SentenceTransformer (all-MiniLM-L12-v2)        │
  │    → embeddings de 384 dimensiones              │
  │    → UMAP (n_components=15, n_neighbors=15,     │
  │            min_dist=0.05, random_state=42)      │
  │    → HDBSCAN (min_cluster_size=20)              │
  │    → One-hot encoding por cluster               │
  │  Clusters resultantes: outdoor, security,       │
  │  balcony_view, entertainment, family, fitness,  │
  │  pet_friendly, accessibility                    │
  ├─────────────────────────────────────────────────┤
  │  Geografía                                      │
  │  Distancia geodésica mínima (km) a 11           │
  │  atracciones turísticas de NYC                  │
  └─────────────────────────────────────────────────┘
        │
        ▼
  [3. Modelado]
  Modelo 1: MLP baseline (solo features estructurales)
  Modelo 2: MLP + amenidades (features estructurales + clusters)
  Modelo 3: Gradient Boosting + GridSearchCV (tuning automático)
        │
        ▼
  [4. Evaluación]
  Métricas: R², MAE, RMSE, MSE
```

---

## Modelos

### Modelo 1 — MLP Baseline (TensorFlow/Keras)

- **Input:** Features estructurales (camas, baños, tipo de habitación, reseñas, ubicación)
- **Arquitectura:** Dense(32, relu) → Dropout(0.3) → Dense(16, relu) → Dropout(0.3) → Dense(8, relu) → Dense(1)
- **Optimizador:** Adam · **Loss:** MSE
- **Objetivo:** Establecer un baseline sin información semántica de amenidades

### Modelo 2 — MLP + Amenidades (TensorFlow/Keras)

- **Input:** Features estructurales + clusters de amenidades (one-hot encoding HDBSCAN)
- **Arquitectura:** Igual que Modelo 1 con mayor dimensión de entrada
- **Diferencia:** Incorpora la información semántica agrupada de las amenidades
- **Técnica clave:** HDBSCAN permite un número variable de clusters sin asumir forma esférica

### Modelo 3 — Gradient Boosting Regressor (scikit-learn)

- **Input:** Features estructurales + clusters de amenidades
- **Tuning:** GridSearchCV (`cv=5`, `n_jobs=-1`)
  - `n_estimators`: range(1, 100, 5) · `learning_rate`: [0.1, 0.01, 0.001] · `max_depth`: [5, 7]
- **Mejores parámetros:** `n_estimators=91`, `learning_rate=0.1`, `max_depth=7`

---

## Resultados

| Modelo | R² | MAE | MSE |
|--------|-----|-----|-----|
| MLP sin amenidades | 0.3823 | 47.49 | 4914.70 |
| MLP + amenidades | 0.5670 | 42.33 | 3445.28 |
| Gradient Boosting | 0.7191 | 0.24 | 3445.28 |

> El Gradient Boosting superó a los modelos MLP en esta tarea tabular. La inclusión de clusters de amenidades aportó una mejora de ~18 puntos en R² respecto al baseline.

---

## Estructura del proyecto

```
DL_Project/
├── Airbnb_Prediction.ipynb       # Notebook principal (133 celdas, 11 secciones)
├── requirements-mac.txt           # Dependencias para macOS
├── requeriments-debian.txt        # Dependencias para Debian/Linux (incluye CUDA)
├── datasets/
│   ├── listings.csv               # Dataset original (no versionado)
│   ├── airbnb_ny_raw.csv          # Limpieza inicial (no versionado)
│   └── Airbnb_NY_cleaned.csv      # Dataset final (no versionado)
├── modelos/
│   ├── mlp_sin_amenidades.keras   # Modelo 1 serializado
│   ├── mlp_con_amenidades.keras   # Modelo 2 serializado
│   ├── gradient_boosting.joblib   # Modelo 3 serializado
│   ├── scaler_standard.joblib     # StandardScaler (modelos 1–3)
│   ├── airbnb_net_pytorch.pt      # State dict PyTorch (Modelo 4)
│   ├── airbnb_net_config.joblib   # Configuración de arquitectura PyTorch
│   ├── umap_reducer.pkl           # UMAP reducer entrenado
│   ├── hdbscan_clusterer.pkl      # HDBSCAN clusterer entrenado
│   └── scaler_pytorch.joblib      # StandardScaler (Modelo 4)
├── utils/
│   ├── funciones.py               # Visualización y análisis exploratorio
│   ├── funciones2.py              # Evaluación de modelos y búsqueda de hiperparámetros
│   └── funciones3.py              # Feature engineering geográfico
├── training_loss.png              # Curvas de entrenamiento del modelo 4
├── model_evaluation.png           # Comparativa de modelos
└── cluster_analysis.png           # Visualización de clusters de amenidades
```

---
## Instalación

### Prerrequisitos

- Python 3.10+
- Jupyter Notebook o JupyterLab

### macOS

```bash
# Crear entorno virtual
python -m venv dl_pro
source dl_pro/bin/activate

# Instalar dependencias
pip install -r requirements-mac.txt
```

### Debian / Ubuntu (con soporte GPU)

```bash
# Crear entorno virtual
python -m venv dl_env
source dl_env/bin/activate

# Instalar dependencias (incluye CUDA, cuDNN, Triton)
pip install -r requeriments-debian.txt
```

> **Nota GPU:** El archivo `requeriments-debian.txt` incluye las dependencias de NVIDIA (CUDA 13.x, cuDNN 9.x) para aceleración GPU con PyTorch. Requiere drivers NVIDIA compatibles instalados en el sistema.

---

## Uso

1. Descargar el dataset desde [Inside Airbnb](https://insideairbnb.com/get-the-data/) (New York City, `listings.csv`) y colocarlo en `datasets/`.

2. Activar el entorno virtual:
   ```bash
   source dl_pro/bin/activate   # macOS
   source dl_env/bin/activate   # Debian
   ```

3. Lanzar Jupyter:
   ```bash
   jupyter notebook Airbnb_Prediction.ipynb
   ```

4. Ejecutar las secciones en orden:
   - **Sección 1–2:** Imports y funciones auxiliares
   - **Sección 3–4:** Carga y limpieza de datos
   - **Sección 5:** Análisis exploratorio (EDA)
   - **Sección 6–7:** Detección de outliers y exportación del dataset limpio
   - **Sección 8:** Preparación para modelado (embeddings, clustering, features geográficas)
   - **Sección 9:** Entrenamiento y evaluación de los 4 modelos
   - **Sección 10:** Resultados finales y persistencia de modelos
   - **Sección 11:** Conclusiones

> La Sección 8 requiere descarga del modelo `all-MiniLM-L12-v2` desde Hugging Face (~120 MB). Se descarga automáticamente la primera vez.

---

## Archivos generados

Al ejecutar el notebook completo se generan los siguientes artefactos:

| Archivo | Descripción |
|---------|-------------|
| `training_loss.png` | Curvas de pérdida (train/validation) del Modelo 4 por época |
| `model_evaluation.png` | Comparativa de métricas (R², MAE, MSE) entre los 4 modelos |
| `cluster_analysis.png` | Visualización UMAP 2D de los clusters de amenidades generados por HDBSCAN |
| `modelos/` | Directorio con todos los modelos serializados listos para producción |

---

## Tecnologías

| Categoría | Librerías |
|-----------|-----------|
| Deep Learning | TensorFlow 2.21, Keras 3.13, PyTorch 2.11 |
| NLP / Embeddings | sentence-transformers 5.3, transformers 5.4 |
| Clustering | HDBSCAN 0.8, UMAP 0.5 |
| ML clásico | scikit-learn 1.8, statsmodels 0.14 |
| Datos | pandas 3.0, numpy 2.4 |
| Visualización | matplotlib 3.10, seaborn 0.13 |
| Geografía | geopy 2.4 |

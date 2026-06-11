# 🧬 Clasificación de Secuencias de ADN mediante Aprendizaje Automático

Este repositorio está orientado a la clasificación de secuencias de ADN mediante técnicas de aprendizaje automático. El proyecto permite entrenar múltiples modelos de clasificación utilizando diferentes métodos de codificación (*encodings*) de secuencias biológicas y posteriormente emplear el mejor modelo para realizar predicciones sobre nuevas secuencias.

---

## 🤖 Modelos Soportados

Actualmente se entrenan los siguientes modelos de aprendizaje automático:

```python
modelos = {
    "KNN": KNeighborsClassifier(),
    "LDA": LinearDiscriminantAnalysis(),
    "LR": LogisticRegression(max_iter=1000, random_state=SEED),
    "NB": GaussianNB(),
    "SVM": SVC(probability=True, random_state=SEED),
    "MLP": MLPClassifier(max_iter=500, random_state=SEED),
    "RF": RandomForestClassifier(
        n_estimators=100,
        random_state=SEED,
        n_jobs=-1
    ),
    "DT": DecisionTreeClassifier(random_state=SEED),
    "GBC": GradientBoostingClassifier(random_state=SEED)
}

if XGBClassifier is not None:
    modelos["XGBoost"] = XGBClassifier(
        random_state=SEED,
        eval_metric="mlogloss"
    )
```

Todos los modelos son entrenados, evaluados y almacenados para su posterior uso en tareas de clasificación de nuevas secuencias.

---

## 📂 Estructura del Proyecto

```text
PROYECTO FINAL/
├── 📁 __pycache__/
├── 📁 data_entrenamiento/                 # Datos de encoding generados
├── 📁 fasta/                              # Archivos FASTA de entrada
├── 📁 figuras/                            # Gráficos de evaluación
│   ├── 📁 matriz_confusion/               # Matrices de confusión
│   └── 📁 ROC/                            # Curvas ROC
├── 📁 modelos_entrenados/                 # Modelos entrenados y mapeo de clases
├── 📁 reportes/                           # Reportes de clasificación
├── 📄 clasificacion_mejor_modelo.py       # Predicción usando el mejor modelo
├── 📄 entrenamiento.py                    # Lógica de entrenamiento
├── 📄 fabrica_encoders.py                 # Gestión de encodings
├── 📄 main.py                             # Script principal
├── 📄 README.md                           # Documentación
├── 📄 resumen_experimentos.csv            # Resumen de resultados
├── 📄 SimpleRepeatSeq_Articulo.pdf        # Artículo de referencia
├── 📄 SimpleRepeatSeq_dataset.fasta_predicciones.csv
│                                          # Ejemplo de salida
└── 📄 utilidades.py                       # Funciones auxiliares
```

---

## 🧬 Métodos de Encoding Soportados

Actualmente se encuentran implementados los siguientes métodos de codificación:

- `eiip`
- `dax`
- `km3`
- `km4`
- `km6`
- `com`

> 💡 **Nota:** Si desea incorporar un nuevo método de encoding, puede hacerlo fácilmente mediante el archivo `fabrica_encoders.py`, diseñado específicamente para extender esta funcionalidad.

---

# 🚀 Instrucciones de Uso

## 1️⃣ Entrenamiento

### Preparación

Ubique los archivos FASTA que desea utilizar dentro de la carpeta:

```text
fasta/
```

Posteriormente ejecute el entrenamiento indicando el archivo FASTA y los encodings que desea utilizar:

```bash
python .\main.py --fasta nombre_archivo.fasta --encoding eiip dax km3 km4 km6 com
```

### Parámetros

| Parámetro | Descripción |
|-----------|-------------|
| `--fasta` | Archivo FASTA utilizado para entrenamiento |
| `--encoding` | Lista de encodings a aplicar |

### Proceso

Durante la ejecución se realizarán las siguientes etapas:

1. Lectura del archivo FASTA.
2. Generación de encodings.
3. Entrenamiento de modelos.
4. Evaluación de desempeño.
5. Generación de reportes y figuras.
6. Almacenamiento de modelos entrenados.

Los datos transformados mediante encoding serán almacenados en:

```text
data_entrenamiento/
```

### ⚠️ Importante

Dentro de la carpeta:

```text
modelos_entrenados/
```

se genera el archivo:

```text
class_mapping.pkl
```

Este archivo contiene el mapeo entre etiquetas originales y clases numéricas utilizadas durante el entrenamiento. Debe conservarse para realizar predicciones correctamente.

---

## 📊 Métricas de Evaluación

Para cada combinación de modelo, encoding y estrategia de procesamiento se almacenan las siguientes métricas:

```python
{
    "Modelo": mod_name,
    "Encoding": enc,
    "Procesamiento": proc_name,
    "Accuracy": acc,
    "Precision": precision,
    "Recall": recall,
    "F1-weighted": f1,
    "ROC-AUC(macro)": auc,
    "ArchivoModelo": nombre_modelo
}
```

### Métricas Generadas

- Accuracy
- Precision
- Recall
- F1-weighted
- ROC-AUC (macro)

---

## 📈 Artefactos Generados

### Curvas ROC

Se almacenan en:

```text
figuras/ROC/
```

### Matrices de Confusión

Se almacenan en:

```text
figuras/matriz_confusion/
```

### Reportes de Clasificación

Se almacenan en:

```text
reportes/
```

Cada reporte contiene el detalle completo del desempeño del modelo evaluado.

---

## 🏆 Selección del Mejor Modelo

Al finalizar el entrenamiento se genera el archivo:

```text
resumen_experimentos.csv
```

Este archivo contiene todos los experimentos ordenados desde el mejor hasta el peor desempeño utilizando los siguientes criterios:

1. ROC-AUC(macro)
2. Accuracy
3. F1-weighted

El mejor modelo se almacena automáticamente como:

```text
modelos_entrenados/best_model.pkl
```

---

## 2️⃣ Predicción (Clasificación con el Mejor Modelo)

Una vez entrenado el sistema, es posible clasificar nuevas secuencias utilizando automáticamente el mejor modelo encontrado.

### Ejecución

```bash
python .\clasificacion_mejor_modelo.py --fasta nombre_archivo.fasta
```

### Funcionamiento

Durante la predicción:

- Se carga automáticamente `best_model.pkl`.
- Se recupera el encoding utilizado por dicho modelo.
- Se aplica la misma estrategia de procesamiento empleada durante el entrenamiento.
- Se generan las predicciones correspondientes.

No es necesario indicar manualmente el encoding ni la estrategia de procesamiento, ya que esta información se obtiene desde:

```text
resumen_experimentos.csv
```

---

## 📄 Resultado de la Predicción

Al finalizar el proceso se genera un archivo CSV con los resultados de clasificación:

```text
nombre_archivo_predicciones.csv
```

Este archivo contiene la información de cada secuencia evaluada junto con la clase predicha por el mejor modelo disponible.

---

## 📚 Referencia

El repositorio incluye el artículo:

```text
SimpleRepeatSeq_Articulo.pdf
```

que sirve como referencia conceptual para el tratamiento y clasificación de secuencias biológicas implementado en este proyecto.

---

## 👨‍🔬 Flujo General del Proyecto

```text
FASTA
  │
  ▼
Encoding
  │
  ▼
Entrenamiento de Modelos
  │
  ▼
Evaluación
  │
  ├── Curvas ROC
  ├── Matrices de Confusión
  ├── Reportes
  ▼
Selección del Mejor Modelo
  │
  ▼
best_model.pkl
  │
  ▼
Predicción de Nuevas Secuencias
```
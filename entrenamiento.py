import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
from xgboost import XGBClassifier
import utilidades

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import shutil
warnings.filterwarnings('ignore')

SEED = 42

def entrenamiento(encodings,n_cls,names):

    def obtener_modelos():
        modelos = {
            "KNN": KNeighborsClassifier(),
            "LDA": LinearDiscriminantAnalysis(),
            "LR": LogisticRegression(max_iter=1000, random_state=SEED),
            "NB": GaussianNB(),
            "SVM": SVC(probability=True, random_state=SEED),
            "MLP": MLPClassifier(max_iter=500, random_state=SEED),
            "RF": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
            "DT": DecisionTreeClassifier(random_state=SEED),
            "GBC": GradientBoostingClassifier(random_state=SEED)
        }
        if XGBClassifier is not None:
            modelos["XGBoost"] = XGBClassifier(random_state=SEED, eval_metric='mlogloss')
        return modelos

    todos_resultados = []

    print("Iniciando experimentos automatizados...\n")

    for enc in encodings:
        try:
            # Cargar X e y para el encoding actual
            X = np.load(f"data_entrenamiento/{enc}.npy")
            y = np.load("data_entrenamiento/y.npy").astype(int)
        except FileNotFoundError:
            print(f"Archivo para encoding '{enc}' no encontrado. Saltando...")
            continue

        cv = StratifiedKFold(
            n_splits=10,
            shuffle=True,
            random_state=SEED
        )

        procesamientos = {
            "Original": [],
            "Scaled": [
                ("scaler", StandardScaler())
            ],
            "PCA": [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.96, random_state=SEED))
            ]
        }

        for proc_name, pasos_pre in procesamientos.items():

            dict_modelos = obtener_modelos()

            for mod_name, model in dict_modelos.items():

                try:

                    pipeline = Pipeline(
                        pasos_pre + [("model", model)]
                    )

                    preds = cross_val_predict(
                        pipeline,
                        X,
                        y,
                        cv=cv,
                        method="predict",
                        n_jobs=-1
                    )

                    probs = cross_val_predict(
                        pipeline,
                        X,
                        y,
                        cv=cv,
                        method="predict_proba",
                        n_jobs=-1
                    )

                    pipeline.fit(X, y)

                    acc = accuracy_score(y, preds)

                    precision = precision_score(
                        y,
                        preds,
                        average="weighted"
                    )

                    recall = recall_score(
                        y,
                        preds,
                        average="weighted"
                    )

                    f1 = f1_score(
                        y,
                        preds,
                        average="weighted"
                    )

                    auc = utilidades.calcular_macro_auc(
                        y,
                        probs
                    )

                    utilidades.plot_roc(
                        y_true=y,
                        y_score=probs,
                        names=names,
                        n_cls=n_cls,
                        archivo=f"{mod_name}_{enc}_{proc_name}_ROC"
                    )

                    utilidades.plot_cm(
                        y_true=y,
                        y_pred=preds,
                        names=names,
                        archivo=f"{mod_name}_{enc}_{proc_name}_CM"
                    )

                    reporte = classification_report(
                        y,
                        preds,
                        target_names=names
                    )

                    with open(
                        f"reportes/{mod_name}_{enc}_{proc_name}.txt",
                        "w",
                        encoding="utf-8"
                    ) as f:
                        f.write(reporte)

                    nombre_modelo = (
                        f"{enc}_{proc_name}_{mod_name}.pkl"
                    )

                    joblib.dump(
                        pipeline,
                        f"./modelos_entrenados/{nombre_modelo}"
                    )

                    todos_resultados.append({
                        "Modelo": mod_name,
                        "Encoding": enc,
                        "Procesamiento": proc_name,
                        "Accuracy": acc,
                        "Precision": precision,
                        "Recall": recall,
                        "F1-weighted": f1,
                        "ROC-AUC(macro)": auc,
                        "ArchivoModelo": nombre_modelo
                    })

                    print(
                        f"{enc} | {proc_name} | {mod_name} "
                        f"| AUC={auc:.4f}"
                    )

                except Exception as e:
                    print(
                        f"Error en modelo {mod_name} "
                        f"con procesamiento {proc_name}: {e}"
                    )

    df_res = pd.DataFrame(todos_resultados)

    df_res_ordenado = (
        df_res
        .sort_values(
            by=[
                "ROC-AUC(macro)",
                "Accuracy",
                "F1-weighted"
            ],
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\n=== TOP 15 EXPERIMENTOS ===")
    print(df_res_ordenado.head(15).to_string())

    df_res_ordenado.to_csv(
        "resumen_experimentos.csv",
        index=False
    )

    print("\nMejor experimento:")
    print(df_res_ordenado.iloc[0])

    mejor_modelo = df_res_ordenado.iloc[0]["ArchivoModelo"]
    shutil.copy2(
        f"./modelos_entrenados/{mejor_modelo}",
        "./modelos_entrenados/best_model.pkl"
    )

    if __name__ == '__main__':
        print("Ejecutado directamente...")
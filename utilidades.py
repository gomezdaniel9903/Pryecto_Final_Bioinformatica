
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing  import  label_binarize
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sn
from sklearn.model_selection   import  learning_curve, StratifiedKFold
from pathlib import Path
import time
DEFAULT_DATO = 1.5

def plot_cm(
    y_true,
    y_pred,
    names,
    title="Matriz de Confusión",
    archivo=""
):

    cm = confusion_matrix(y_true, y_pred)

    df = pd.DataFrame(
        cm,
        index=names,
        columns=names
    )

    plt.figure(figsize=(6, 5))

    sn.heatmap(
        df,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5
    )

    plt.title(title)
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.tight_layout()

    plt.savefig(
        f"figuras/matriz_confusion/{archivo}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def plot_roc(y_true, y_score, names, n_cls,
             title='Curvas ROC',
             archivo="roc.png"):

    plt.figure(figsize=(7,5))

    # Caso binario
    if n_cls == 2:

        fpr, tpr, _ = roc_curve(y_true, y_score[:,1])

        plt.plot(
            fpr,
            tpr,
            lw=2,
            label=f'{names[1]} AUC={auc(fpr,tpr):.3f}'
        )

    # Caso multiclase
    else:

        y_bin = label_binarize(
            y_true,
            classes=list(range(n_cls))
        )

        colors = ['#1C8FC4','#46A832','#D63B8A','#F7841C']

        for i, (name, col) in enumerate(zip(names, colors)):

            fpr, tpr, _ = roc_curve(
                y_bin[:, i],
                y_score[:, i]
            )

            plt.plot(
                fpr,
                tpr,
                color=col,
                lw=2,
                label=f'{name} AUC={auc(fpr,tpr):.3f}'
            )

    plt.plot([0,1],[0,1],'k--',lw=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)

    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.savefig(
        f"figuras/ROC/{archivo}.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

def calcular_macro_auc(y_true, y_prob):

        if len(np.unique(y_true)) == 2:
            return roc_auc_score(y_true, y_prob[:, 1])

        y_bin = label_binarize(
            y_true,
            classes=np.unique(y_true)
        )

        return roc_auc_score(
            y_bin,
            y_prob,
            average="macro",
            multi_class="ovr"
        )

def plot_lc_sklearn(model, X, y, title='Curva de Aprendizaje',archivo=""):
    sizes, tr_sc, va_sc = learning_curve(
        model, X, y, cv=StratifiedKFold(5), scoring='accuracy',
        train_sizes=np.linspace(0.1,1.0,8), n_jobs=-1)
    tr_m, va_m = tr_sc.mean(1), va_sc.mean(1)
    tr_s, va_s = tr_sc.std(1),  va_sc.std(1)
    plt.figure(figsize=(7,4))
    plt.plot(sizes, tr_m, 'o-', label='Train');   plt.fill_between(sizes,tr_m-tr_s,tr_m+tr_s,alpha=.15)
    plt.plot(sizes, va_m, 's-', label='Val');     plt.fill_between(sizes,va_m-va_s,va_m+va_s,alpha=.15)
    plt.xlabel('Muestras'); plt.ylabel('Accuracy'); plt.title(title)
    plt.legend(); plt.tight_layout()
    plt.savefig(f"figuras/{archivo}", dpi=300, bbox_inches='tight')

def mapeo_clases_unicas(clases):
    map_clases = {}
    for i in range(len(clases)):
        map_clases[clases[i]] = i
    return map_clases

def generar_archivo(tipo_encoder,codificacion):
    max_len = max(len(x) for x in codificacion)

    X = np.array([
        np.pad(
            x,
            (0, max_len - len(x)),
            constant_values=DEFAULT_DATO
        )
        for x in codificacion
    ])

    np.save(f"data_entrenamiento/{tipo_encoder}.npy", X)

def generar_archivo_y(codificacion):
    np.save("data_entrenamiento/y.npy", np.array(codificacion))
    return

def esperar_archivo(nombre_archivo):
    ruta = f"data_entrenamiento/{nombre_archivo}"
    file_path = Path(ruta)
    if file_path.is_file():
        print(f"Archivo {ruta} creado correctamente.")
        return True
    time.sleep(5)
    return False

if __name__ == '__main__':
    print("Ejecutado directamente...")
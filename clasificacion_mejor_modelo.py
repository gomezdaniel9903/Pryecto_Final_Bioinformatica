import joblib
import utilidades
from Bio import SeqIO
from fabrica_encoders import FabricaEncoders
import argparse
import numpy as np
import pandas as pd
if __name__ == '__main__':

    model = joblib.load(
    "./modelos_entrenados/best_model.pkl"
    )

    info = pd.read_csv(
        "./resumen_experimentos.csv",
        nrows=1
    )

    encoding_modelo = info.iloc[0]["Encoding"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)

    args = parser.parse_args()
    fasta = args.fasta
    seqs = list(
        SeqIO.parse(
            f"fasta/{fasta}",
            "fasta"
        )
    )

    encoder = FabricaEncoders(
        tipo_encoder=encoding_modelo,
        seqs=seqs
    )

    codificacion = encoder.seleccion_encoder()
    nombre_archivo = f"{encoding_modelo}_best_model"
    utilidades.generar_archivo(
        nombre_archivo,
        codificacion
    )

    archivo_creado = False
    while(archivo_creado == False):
        archivo_creado = utilidades.esperar_archivo(f"{nombre_archivo}.npy")

    X = np.load(
        f"data_entrenamiento/{encoding_modelo}_best_model.npy"
    )

    preds = model.predict(X)
    probs = model.predict_proba(X)

    map_clases = joblib.load(
        "modelos_entrenados/class_mapping.pkl"
    )

    map_clases_inv = {
    v: k for k, v in map_clases.items()
    }

    resultados = []

    for seq, pred, prob in zip(seqs, preds, probs):

        resultados.append({
            "ID": seq.id,
            "ClasePredicha": map_clases_inv[pred],
            "Probabilidad": float(prob[pred])
        })

    pd.DataFrame(resultados).to_csv(
        f"{fasta}_predicciones.csv",
        index=False
    )

    print(f"{fasta}_predicciones.csv generado con los resultados de la clasificación.")
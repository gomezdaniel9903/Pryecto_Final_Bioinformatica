from Bio import SeqIO
from fabrica_encoders import FabricaEncoders
import utilidades
import argparse
import entrenamiento
import numpy as np
import joblib
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--encoding",
        nargs="+",
        required=True
    )
    parser.add_argument("--fasta", required=True)
    args = parser.parse_args()
    fasta = args.fasta
    seqs = list(SeqIO.parse(f"fasta/{fasta}","fasta"))
    clases = sorted(set(seq.id.split("#")[1] for seq in seqs))
    map_clases = utilidades.mapeo_clases_unicas(list(clases))
    joblib.dump(
        map_clases,
        "./modelos_entrenados/class_mapping.pkl"
    )
    codificacion_id = [map_clases[seq.id.split("#")[1]] for seq in seqs]

    
    tipo_encoders = args.encoding
    print(np.array(codificacion_id).shape)
    utilidades.generar_archivo_y(codificacion_id)
    archivo_creado = False
    while(archivo_creado == False):
        archivo_creado = utilidades.esperar_archivo("y.npy")
    for tipo_encoder in tipo_encoders:
        encoder = FabricaEncoders(tipo_encoder=tipo_encoder,seqs=seqs)
        codificacion = encoder.seleccion_encoder()
        utilidades.generar_archivo(tipo_encoder, codificacion)
        archivo_creado = False
        nombre_archivo = f"{tipo_encoder}.npy"
        while(archivo_creado == False):
            archivo_creado = utilidades.esperar_archivo(f"{tipo_encoder}.npy")
    print("Iniciando entrenamiento...")
    entrenamiento.entrenamiento(tipo_encoders,len(clases),list(clases))


        
        
    
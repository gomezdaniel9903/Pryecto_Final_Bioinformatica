import numpy as np


DEFAULT_DATO = 1.5
class FabricaEncoders:
    def __init__(self, tipo_encoder,seqs):
        super().__init__()
        self.tipo_encoder =  tipo_encoder
        self.seqs = seqs
        

    def seleccion_encoder(self):
        if(self.tipo_encoder == 'km3'):
            return self.kmers(3)
        elif(self.tipo_encoder == 'km4'):
            return self.kmers(4)
        elif(self.tipo_encoder == 'km6'):
            return self.kmers(6)
        elif(self.tipo_encoder == 'dax'):
            return self.dax()
        elif(self.tipo_encoder == 'eiip'):
            return self.eiip()
        elif(self.tipo_encoder == 'com'):
            return self.com()

    @staticmethod
    def kmers_encoder(num_kmers = 3,secuencia=None):
        resultado = []
        secuencia_str = str(secuencia).upper()
        mapa_kmer = {"A": 2, "T": 1, "G": 3, "C": 0, "N": 1.5}

        for i in range(len(secuencia_str) - num_kmers + 1):
            kmer = secuencia_str[i:i+num_kmers]
            codificado = [mapa_kmer.get(x, DEFAULT_DATO) for x in kmer]
            resultado.append(codificado)
        return np.array(resultado).flatten()
    
    def kmers(self,num_kmers):
        codificacion = []
        for seq in self.seqs:
            codificacion.append(self.kmers_encoder(num_kmers,secuencia=seq))
            
        return codificacion
    
    def dax(self):
        codificacion_DAX = []
        mapa_dax = {"A": 2, "T": 1, "G": 3, "C": 0}
        for seq in self.seqs:
            codificacion_DAX.append([mapa_dax.get(x, DEFAULT_DATO) for x in seq.seq.upper()])
        return codificacion_DAX
    
    def eiip(self):
        codificacion_eiip = []
        mapa_eiip = {"A": 0.1260, "T": 0.1335, "G": 0.0806, "C": 0.1340, "N": 1.5}
        for seq in self.seqs:
            codificacion_eiip.append([mapa_eiip.get(x, DEFAULT_DATO) for x in seq.seq.upper()])
        return codificacion_eiip
    
    def com(self):
        codificacion_complementary = []
        mapa_complementary = {"A": 2, "T": -2, "G": 1, "C": -1, "N": 1.5}
        for seq in self.seqs:
            codificacion_complementary.append([mapa_complementary.get(x, DEFAULT_DATO) for x in seq.seq.upper()])
        return codificacion_complementary
        
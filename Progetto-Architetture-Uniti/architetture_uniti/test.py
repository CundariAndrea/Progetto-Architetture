import argparse
import sys
import os
import subprocess
import time
import numpy as np
import struct
from pathlib import Path


import gruppo23

try:
    import gruppo23.quantpivot64
    import gruppo23.quantpivot64omp
except ImportError:
    pass 

class QuantPivot32Process:
    def __init__(self, exe_path="src/32/knn"):
        self.exe_path = exe_path
        if not os.path.exists(self.exe_path): 
            base_path = os.path.dirname(os.path.abspath(__file__))
            self.exe_path = os.path.join(base_path, "src", "32", "knn")
            if not os.path.exists(self.exe_path):
                raise FileNotFoundError(f"Eseguibile 32-bit non trovato in: {self.exe_path}. Hai fatto 'make build' in src/32?")

    def fit(self, dataset, h, x, silent):
        self.h = h
        self.x = x
        pass

    def predict(self, query, k, silent):
        out_ids = "results_ids_2000x8_k8_x64_32.ds2"
        out_dists = "results_dst_2000x8_k8_x64_32.ds2"
        
      
        ds_file = args.DS
        q_file = args.Q

        cmd = [
            self.exe_path,
            "-d", ds_file,
            "-q", q_file,
            "-i", out_ids,
            "-e", out_dists,
            "-h", str(self.h),
            "-k", str(k),
            "-x", str(self.x)
        ]
        
        if silent:
            cmd.append("-s")

        print(f"[Wrapper 32-bit] Esecuzione: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        ids_res = load_ds2(out_ids, dtype='float32')
        dists_res = load_ds2(out_dists, dtype='float32')
        
     
        if os.path.exists(out_ids): os.remove(out_ids)
        if os.path.exists(out_dists): os.remove(out_dists)
        
        return ids_res, dists_res



def load_ds2(filename, dtype, alignment=None):
    with open(filename, 'rb') as f:
        header = f.read(8)
        n, d = struct.unpack('ii', header)
        
        data = np.frombuffer(f.read(), dtype=dtype)
        data = data.reshape((n, d))
    return data

def save_ds2(data, ds2name, dtype):
    n, d = data.shape
    with open(ds2name, 'wb') as f:
        f.write(struct.pack('ii', n, d))
        data.astype(dtype).tofile(f)

def csv_to_ds2(csvname, dtype, delimiter=','):
    path = Path(csvname)
    name = path.stem
    ds2name = f"{name}.ds2"
    data = np.loadtxt(csvname, delimiter=',').astype(dtype)
    n, d = data.shape
    with open(ds2name, 'wb') as f:
        f.write(struct.pack('ii', n, d))
        data.tofile(f)

def load_file(file_name, dtype, alignment=None):
    path = Path(file_name)
    name = path.stem
    ext = path.suffix
    if ext.lower() not in ['.csv', '.ds2']:
        raise Exception("Formato file non riconosciuto")
    if ext == ".csv":
        if not Path(f"{name}.ds2").is_file():
            csv_to_ds2(file_name, dtype)
        file_name = f"{name}.ds2"
    return load_ds2(file_name, dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test del progetto QuantPivot')

    parser.add_argument('DS', help='nome file dataset')
    parser.add_argument('Q', help='nome file query')
    parser.add_argument('h', type=int, help='numero di pivot')
    parser.add_argument('k', type=int, help='numero di vicini')
    parser.add_argument('x', type=int, help='parametro di quantizzazione')
    parser.add_argument('t', type=str, choices=['32', '64', '64omp'], help='Versione')
    parser.add_argument('-s', '--silent', action='store_true', help='modalità silenziosa')

    global args 
    args = parser.parse_args()
    if(not args.silent): print(args)

    
    bits = 32 if args.t == '32' else 64 
    dtype_np = 'float32' if bits == 32 else 'float64'

    if not os.path.exists(args.DS): sys.exit(f"Errore: {args.DS} non esiste")
    if not os.path.exists(args.Q): sys.exit(f"Errore: {args.Q} non esiste")

  
    print(f"Caricamento dati ({dtype_np})...")
    DS = load_file(args.DS, dtype=dtype_np)
    Q = load_file(args.Q, dtype=dtype_np)

    if args.t == '32':
        print("--- Modalità 32-bit (Processo Esterno) ---")
        quantpivot = QuantPivot32Process() 
    elif args.t == '64':
        print("--- Modalità 64-bit (Libreria Interna) ---")
        quantpivot = gruppo23.quantpivot64._quantpivot64.QuantPivot()
    elif args.t == '64omp':
        print("--- Modalità 64-bit OMP (Libreria Interna) ---")
        quantpivot = gruppo23.quantpivot64omp._quantpivot64omp.QuantPivot()

    print("Avvio FIT...")
    start = time.time()
    quantpivot.fit(DS, args.h, args.x, args.silent)
    fit_time = time.time() - start

    print("Avvio PREDICT...")
    start = time.time()
    ids, dists = quantpivot.predict(Q, args.k, args.silent)
    prd_time = time.time() - start

    if not args.silent:
        print(f"FIT time: {fit_time:.5f} s")
        print(f"PRD time: {prd_time:.5f} s")
    else:
        print(f"{fit_time} {prd_time}")

   
    save_ds2(ids, f"idNN_{args.t}.ds2", dtype=dtype_np)
    save_ds2(dists, f"distNN_{args.t}.ds2", dtype=dtype_np)
import gruppoX
import gruppoX.quantpivot32

import argparse
import sys
import os
import time
import numpy as np
import numpy.random
import struct
import pyfftw
from pathlib import Path

def load_ds2(filename, dtype, alignment):
    with open(filename, 'rb') as f:
        # Leggi header (n, d)
        header = f.read(8)  # 4 + 4 byte
        n, d = struct.unpack('ii', header)  # 'ii' = 2 int32
        # Alloca array allineato
        aligned = pyfftw.empty_aligned((n, d), dtype=dtype, n=alignment)
        # Leggi dati direttamente nell'array allineato
        bytes_read = f.readinto(aligned.data)
    return aligned

def save_ds2(data, ds2name, dtype):
    n, d = data.shape
    with open(ds2name, 'wb') as f:
        f.write(struct.pack('ii', n, d))
        data.tofile(f)

def csv_to_ds2(csvname, dtype, delimiter=','):
    path = Path(csvname)
    name = path.stem
    ext = path.suffix
    ds2name = f"{name}.ds2"
    data = np.loadtxt(csvname, delimiter=',').astype(dtype)
    n, d = data.shape
    with open(ds2name, 'wb') as f:
        f.write(struct.pack('ii', n, d))
        data.tofile(f)

def load_file(file_name, dtype, alignment, delimiter=','):
    path = Path(file_name)
    name = path.stem
    ext = path.suffix
    if ext.lower() not in ['.csv', '.ds2']:
        raise("Formato file non riconosciuto")
    if ext == ".csv":
        if not Path(f"{name}.ds2").is_file():
            csv_to_ds2(file_name, dtype)
        file_name = f"{name}.ds2"
    return load_ds2(file_name, dtype, alignment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test del progetto QuantPivot')

    # Definizione dei parametri
    parser.add_argument('DS', help='nome file dataset (csv o ds2)')
    parser.add_argument('Q', help='nome file query (csv o ds2)')
    parser.add_argument('h', type=int, help='numero di pivot')
    parser.add_argument('k', type=int, help='numero di vicini')
    parser.add_argument('x', type=int, help='parametro di quantizzazione')
    parser.add_argument('t', type=str, choices=['32', '64', '64omp'], help='float+sse, double+avx, double+avx+openmp')
    parser.add_argument('-s', '--silent', action='store_true', help='modalità silenziosa')

    # Parsing degli argomenti
    args = parser.parse_args()
    if(not args.silent): print(args)

    bits = int(args.t.replace('omp',''))

    # Validazione file DS
    if not os.path.exists(args.DS):
        print(f"Errore: Il file {args.DS} non esiste")
        sys.exit(1)

    ds_path = Path(args.DS)
    if ds_path.suffix.lower() not in ['.csv', '.ds2']:
        print(f"Errore: Il file {args.DS} deve avere estensione .csv o .ds2")
        sys.exit(1)

    # Validazione file Q
    if not os.path.exists(args.Q):
        print(f"Errore: Il file {args.Q} non esiste")
        sys.exit(1)

    q_path = Path(args.Q)
    if q_path.suffix.lower() not in ['.csv', '.ds2']:
        print(f"Errore: Il file {args.Q} deve avere estensione .csv o .ds2")
        sys.exit(1)

    # Validazione parametri numerici
    if args.h <= 0:
        print("Errore: h deve essere un intero positivo")
        sys.exit(1)

    if args.k <= 0:
        print("Errore: k deve essere un intero positivo")
        sys.exit(1)

    if args.x <= 0:
        print("Errore: x deve essere un intero positivo")
        sys.exit(1)

    DS = load_file(args.DS, dtype=f'float{bits}', alignment=int(bits/2))
    Q = load_file(args.Q, dtype=f'float{bits}', alignment=int(bits/2))

    if args.t == '32':
        quantpivot = gruppoX.quantpivot32.QuantPivot()
    elif args.t == '64':
        quantpivot = gruppoX.quantpivot64.QuantPivot()
    elif args.t == '64omp':
        quantpivot = gruppoX.quantpivot64omp.QuantPivot()

    # =========================
    start = time.time()
    quantpivot.fit(DS, args.h, args.x, args.silent)
    fit_time = time.time() - start
    # =========================

    # =========================
    start = time.time()
    ids, dists = quantpivot.predict(Q, args.k, args.silent)
    prd_time = time.time() - start
    # =========================

    if not args.silent:
        print("ID NN:")
        print(ids)
        print("DIST NN:")
        print(dists)

    if not args.silent:
        print(f"FIT time: {fit_time:.5f} seconds")
        print(f"PRD time: {prd_time:.5f} seconds")
    else:
        print(fit_time)
        print(prd_time)

    save_ds2(ids, f"idNN_{args.t}_size-{DS.shape[0]}x{DS.shape[1]}_nq-{Q.shape[0]}.ds2", dtype = f'float{args.t}')

    save_ds2(ids, f"distNN_{args.t}_size-{DS.shape[0]}x{DS.shape[1]}_nq-{Q.shape[0]}.ds2", dtype = f'float{args.t}')

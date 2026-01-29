# Makefile aggiornato per la cartella 'architetture_uniti'

# Percorsi relativi ai dataset (siamo in architetture_uniti, datasets è un livello sopra)
DS_DIR = ../datasets
DS_64 = $(DS_DIR)/dataset_2000x256_64.ds2
Q_64  = $(DS_DIR)/query_2000x256_64.ds2
DS_32 = $(DS_DIR)/dataset_2000x256_32.ds2
Q_32  = $(DS_DIR)/query_2000x256_32.ds2

PARAMS = 16 8 64

.PHONY: install test test_omp test32 clean

#install setup.py
install:
	pip install ./ProgettoGruppo23

# 2. Test 64-bit
test: install
	python3 test.py $(DS_64) $(Q_64) $(PARAMS) 64

# 3. Test 64-bit OpenMP
test_omp: install
	python3 test.py $(DS_64) $(Q_64) $(PARAMS) 64omp

# 4. Test 32-bit
test32:
	# Compila il C 32-bit (percorso aggiornato alla sottocartella)
	$(MAKE) -C ProgettoGruppo23/src/32 build
	python3 test.py $(DS_32) $(Q_32) $(PARAMS) 32

clean:
	pip uninstall -y gruppo23
	$(MAKE) -C ProgettoGruppo23/src/32 clean
	rm -rf ProgettoGruppo23/build ProgettoGruppo23/*.egg-info ProgettoGruppo23/src/64/*.o ProgettoGruppo23/src/64omp/*.o
	rm -f *.ds2
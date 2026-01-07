%include "sseutils64.nasm"

default rel

section .data			; Sezione contenente dati inizializzati

input		equ		8
msg			db	'nq:',32,0
nl			db	10,0


section .bss			; Sezione contenente dati non inizializzati
	alignb 16
	nq		resd		1

section .text			; Sezione contenente il codice macchina


; ----------------------------------------------------------
; macro per l'allocazione dinamica della memoria
;
;	getmem	<size>,<elements>
;
; alloca un'area di memoria di <size>*<elements> bytes
; (allineata a 16 bytes) e restituisce in EAX
; l'indirizzo del primo bytes del blocco allocato
; (funziona mediante chiamata a funzione C, per cui
; altri registri potrebbero essere modificati)
;
;	fremem	<address>
;
; dealloca l'area di memoria che ha inizio dall'indirizzo
; <address> precedentemente allocata con getmem
; (funziona mediante chiamata a funzione C, per cui
; altri registri potrebbero essere modificati)

extern get_block
extern free_block

%macro	getmem	2
	mov	eax, %1
	push	eax
	mov	eax, %2
	push	eax
	call	get_block wrt ..plt    ; Per PIC
	add	esp, 8
%endmacro

%macro	fremem	1
	push	%1
	call	free_block wrt ..plt    ; Per PIC
	add	esp, 4
%endmacro

; ------------------------------------------------------------
; Funzioni
; ------------------------------------------------------------

global prova


prova:
		; ------------------------------------------------------------
		; Sequenza di ingresso nella funzione
		; ------------------------------------------------------------
		push		rbp		; salva il Base Pointer
		mov		rbp, rsp	; il Base Pointer punta al Record di Attivazione corrente
		push		rbx		; salva i registri da preservare
		push		rsi
		push		rdi
		; ------------------------------------------------------------
		; legge i parametri dal Record di Attivazione corrente
		; ------------------------------------------------------------

		; elaborazione
		; [RDI] input->DS; 			// dataset
		; [RDI+8]input->P;			// vettore contenente gli indici dei pivot
		; [RDI+16]input->index;		// indice
		; [RDI+24]input->Q;			// query
		; [RDI+32]input->id_nn;		// per ogni query point gli ID dei K-NN
		; [RDI+40]input->dist_nn;	// per ogni query point le distanze dai K-NN
		; [RDI+48]input->h;			// numero di pivot
		; [RDI+52]input->k;			// numero di vicini
		; [RDI+56]input->x;			// parametro x per la quantizzazione
		; [RDI+60]input->N;			// numero di righe del dataset
		; [RDI+64]input->D;			// numero di colonne/feature del dataset
		; [RDI+68]input->nq;		// numero delle query
		; [RDI+72]input->silent;	// modalità silenziosa

		; STAMPA IL PARAMETRO nq
		VMOVSS XMM0, [RDI+68]
		VMOVSS [nq], XMM0
		prints msg
		printsi nq
		prints nl

		; SALVA 7 COME PRIMO INDICE DEI VICINI
		MOV RAX, [RDI+32] ; indirizzo di id_nn
		mov [RAX], dword 15

		; ------------------------------------------------------------
		; Sequenza di uscita dalla funzione
		; ------------------------------------------------------------

		pop	rdi		; ripristina i registri da preservare
		pop	rsi
		pop	rbx
		mov	rsp, rbp	; ripristina lo Stack Pointer
		pop	rbp		; ripristina il Base Pointer
		ret			; torna alla funzione C chiamante

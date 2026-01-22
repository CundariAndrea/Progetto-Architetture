; --------------------------------------------------
; Macro per il linking di file oggetto
; NASM con la C standard library e
; alcune utility per l'uso di SSE
; VERSIONE PIC-SAFE per shared library
;
; Uso:
;	%include "sseutils32.nasm"
;	(da specificare nel file sorgente NASM
;	prima della section .data)
; --------------------------------------------------

default rel

extern	printf

section	.bss

dbuf:	resq	1

section	.data

dmask:	db		'%f',0
imask:	db		'%i',0
cr:		db		10,0
br1:	db		'( ',0
br2:	db		')',10,0
space:	db		32,0

%macro	start	0
		push	rbp
		mov		rbp, rsp
		pushaq
%endmacro

%macro	stop	0
		popaq
		mov		rsp, rbp
		pop		rbp
		ret
%endmacro

%macro	pushaq	0
	push	rax
	push	rbx
	push	rcx
	push	rdx
	push	rsi
	push	rdi
	push	r8
	push	r9
	push	r10
	push	r11
	push	r12
	push	r13
	push	r14
	push	r15
%endmacro

%macro	popaq	0
	pop		r15
	pop		r14
	pop		r13
	pop		r12
	pop		r11
	pop		r10
	pop		r9
	pop		r8
	pop		rdi
	pop		rsi
	pop		rdx
	pop		rcx
	pop		rbx
	pop		rax
%endmacro

; Versione PIC-safe: usa LEA invece di PUSH diretto
%macro	prints	1
		pushaq
		; Allinea stack a 16 byte per chiamata
		push	rbp
		mov		rbp, rsp
		and		rsp, -16

		; Carica indirizzo con LEA (PIC-safe)
		lea		rdi, [%1]
		xor		rax, rax
		call	printf wrt ..plt

		mov		rsp, rbp
		pop		rbp
		popaq
%endmacro

; Versione PIC-safe con LEA
%macro	dprint	1
		pushaq
		push	rbp
		mov		rbp, rsp
		and		rsp, -16

		; Carica il valore double da stampare
		movsd	xmm0, [%1]
		lea		rdi, [dmask]
		mov		rax, 1		; 1 argomento floating point in xmm0
		call	printf wrt ..plt

		mov		rsp, rbp
		pop		rbp
		popaq
%endmacro

; Versione PIC-safe per interi
%macro	printsi	1
		push	rbp
		mov		rbp, rsp
		and		rsp, -16

		push	rsi        ; salva RSI
		push	rdi        ; salva RDI
		push	rax        ; salva RAX

		mov		esi, dword [%1]    ; Legge int32 da memoria
		lea		rdi, [imask]
		xor		rax, rax
		call	printf wrt ..plt

		pop		rax
		pop		rdi
		pop		rsi

		mov		rsp, rbp
		pop		rbp
%endmacro

%macro printpi 2
	; %1 = indirizzo base dell'array
	; %2 = numero di elementi N
	push	rbp
	mov		rbp, rsp
	push	rbx							; salva registri che verranno modificati
	push	r12
	push	r13
	push	r14
	push	rsi
	push	rdi
	push	rax

	mov		r12, %1						; r12 = indirizzo base array
	mov		r13, %2						; r13 = contatore N
	shl		r13, 2						; r13 = r13 x 4
	mov		r14, 0						; r14 = indice corrente

	prints	br1
	%%print_loop:
	cmp		r14, r13					; controlla se abbiamo stampato tutti gli elementi
	jge		%%end_print

	; Stampa l'elemento corrente
	mov		esi, dword [r12 + r14*4]	; carica intero (4 byte per int)
	lea		rdi, [imask]
	xor		eax, eax
	call	printf wrt ..plt
	inc		r14							; incrementa indice
	prints	space
	jmp		%%print_loop

	%%end_print:
	prints	br2

	pop		rax
	pop		rdi
	pop		rsi

	pop     r14							; ripristina registri
	pop     r13
	pop     r12
	pop     rbx
	mov     rsp, rbp
	pop     rbp
%endmacro

%macro	sprint	1
		finit
		fld		dword [%1]
		fst		qword [dbuf]
		dprint	dbuf
%endmacro

%macro	printss	1
		sprint	%1
		prints	cr
%endmacro

%macro	printps	2
		prints	br1
		push	rdx
		push	rcx
		mov		rdx, %1
		mov		rcx, %2
%%loopps:
		sprint	rdx
		sprint	rdx+4
		sprint	rdx+8
		sprint	rdx+12
		add		rdx, 16
		dec		rcx
		jnz		%%loopps
		pop		rcx
		pop		rdx
		prints	br2
%endmacro

%macro	vprintps	2
		prints	br1
		push	rdx
		push	rcx
		mov		rdx, %1
		mov		rcx, %2
%%loopps:
		sprint	rdx
		sprint	rdx+4
		sprint	rdx+8
		sprint	rdx+12
		sprint	rdx+16
		sprint	rdx+20
		sprint	rdx+24
		sprint	rdx+28
		add		rdx, 32
		dec		rcx
		jnz		%%loopps
		pop		rcx
		pop		rdx
		prints	br2
%endmacro

%macro	printsd	1
		dprint	%1
		prints	cr
%endmacro

%macro	printpd	2
		prints	br1
		push	rdx
		push	rcx
		mov		rdx, %1
		mov		rcx, %2
%%looppd:
		dprint	rdx
		dprint	rdx+8
		add		rdx, 16
		dec		rcx
		jnz		%%looppd
		pop		rcx
		pop		rdx
		prints	br2
%endmacro

%macro	vprintpd	2
		prints	br1
		push	rdx
		push	rcx
		mov		rdx, %1
		mov		rcx, %2
%%looppd:
		dprint	rdx
		dprint	rdx+8
		dprint	rdx+16
		dprint	rdx+24
		add		rdx, 32
		dec		rcx
		jnz		%%looppd
		pop		rcx
		pop		rdx
		prints	br2
%endmacro

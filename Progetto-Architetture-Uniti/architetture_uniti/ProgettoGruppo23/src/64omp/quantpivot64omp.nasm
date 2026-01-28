%include "sseutils64.nasm"

default rel             ; FONDAMENTALE per -fPIC: usa indirizzi relativi

section .data
    input       equ     8
    msg         db  'nq:',32,0
    nl          db  10,0

section .bss
    alignb 16
    nq      resd        1

section .text

extern get_block
extern free_block

; ----------------------------------------------------------
; MACRO CORRETTE PER 64-BIT (System V ABI)
; ----------------------------------------------------------
; Gli argomenti si passano nei registri:
; 1° arg: RDI
; 2° arg: RSI
; 3° arg: RDX
; ...
; ----------------------------------------------------------

%macro  getmem  2
    ; Argomento 1 (size) in RDI
    mov     rdi, %1
    ; Argomento 2 (elements) in RSI
    mov     rsi, %2
    ; Chiamata a funzione (rispetta la PLT per -fPIC)
    call    get_block wrt ..plt
    ; Il risultato è già in RAX, non serve pulire lo stack (add esp, 8 NON SI FA in x64)
%endmacro

%macro  fremem  1
    ; Argomento 1 (address) in RDI
    mov     rdi, %1
    call    free_block wrt ..plt
%endmacro

; ------------------------------------------------------------
; Funzioni
; ------------------------------------------------------------

global prova

prova:
        ; ------------------------------------------------------------
        ; Prologo Standard 64-bit
        ; ------------------------------------------------------------
        push    rbp
        mov     rbp, rsp
        push    rbx
        push    r12
        push    r13
        push    r14
        push    r15
        ; (RDI e RSI non sono callee-saved, non è obbligatorio pusharli
        ; a meno che tu non voglia preservarli per dopo)

        ; ------------------------------------------------------------
        ; Corpo della funzione
        ; ------------------------------------------------------------

        ; RDI contiene già il puntatore a params* input

        ; Salva nq (offset 68) in variabile locale
        ; Usa un registro temporaneo per sicurezza
        mov     eax, [rdi + 68]
        mov     [nq], eax

        ; Stampa messaggio 'nq:'
        ; Se 'prints' è una macro di sseutils, speriamo usi LEA o supporti REL.
        ; Se ti dà ancora errore su 'prints msg', prova a commentarlo.
        prints  msg
        
        ; Stampa valore intero di nq
        printsi nq
        
        ; Stampa a capo
        prints  nl

        ; Modifica id_nn (offset 32)
        mov     rax, [rdi + 32]     ; Carica indirizzo puntatore id_nn
        mov     dword [rax], 15     ; Scrive 15 nella prima cella

        ; ------------------------------------------------------------
        ; Epilogo
        ; ------------------------------------------------------------
        pop     r15
        pop     r14
        pop     r13
        pop     r12
        pop     rbx
        mov     rsp, rbp
        pop     rbp
        ret
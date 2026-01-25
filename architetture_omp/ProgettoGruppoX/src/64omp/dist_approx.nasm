; -----------------------------------------------------------------------------
; Funzione dist_approx ULTRA-OPTIMIZED O(x^2) AVX (YMM 256-bit)
;
; Ottimizzazioni rispetto alla versione precedente:
; 1. XOR Logico sui Double: Sostituisce la catena (Sub -> Abs) sfruttando
;    la rappresentazione binaria di 0.0 e 1.0. Risparmia cicli critici.
; 2. Pipeline snellita: Meno dipendenze nel ciclo interno.
;
; Istruzioni utilizzate (Tutte Standard AVX Floating Point):
; - vmovd (Int Load), vcvtdq2pd (Convert), vbroadcastsd (Duplicate)
; - vcmppd (Compare), vxorpd (XOR - Logic), vmulpd (Mult), vsubpd (Sub)
; - vandpd (Mask), vaddpd (Add)
; -----------------------------------------------------------------------------

global dist_approx

section .data
    align 32
    const_1:    dq 1.0, 1.0, 1.0, 1.0   ; Vettore di 1.0
    const_2:    dq 2.0, 2.0, 2.0, 2.0   ; Vettore di 2.0
    ; Nota: Non serve più la maschera abs_mask grazie all'ottimizzazione XOR

section .text

dist_approx:
    push    rbp
    mov     rbp, rsp

    ; --- Setup Costanti ---
    vmovapd ymm10, [const_1]        ; YMM10 = [1.0, 1.0, 1.0, 1.0]
    vmovapd ymm11, [const_2]        ; YMM11 = [2.0, 2.0, 2.0, 2.0]
    
    vxorpd  ymm0, ymm0, ymm0        ; YMM0 = Accumulatore Totale (azzeralo)

    xor     r9, r9                  ; i = 0

.outer_loop:
    cmp     r9d, r8d                ; if i >= x
    jge     .end_loops

    ; --- PREPARAZIONE DATI VETTORE V (Broadcast) ---
    
    ; 1. Carica v_idx[i] (1 intero) e converti in double
    vmovd   xmm1, dword [rdi + r9*4] ; Carica 32 bit (int) nella parte bassa
    vcvtdq2pd xmm1, xmm1             ; Converte int -> double (Low qword corretto)
    vbroadcastsd ymm1, xmm1          ; YMM1 = [v_idx[i] x 4]

    ; 2. Carica v_sign[i] e converti
    vmovd   xmm2, dword [rsi + r9*4]
    vcvtdq2pd xmm2, xmm2
    vbroadcastsd ymm2, xmm2          ; YMM2 = [v_sign[i] x 4]

    xor     r10, r10                 ; j = 0

.inner_loop:
    cmp     r10d, r8d
    jge     .next_outer

    ; --- CORE LOOP (4 Operazioni alla volta) ---

    ; 3. Carica e converti 4 elementi di W_IDX
    vmovdqu xmm3, [rdx + r10*4]      ; Carica 4 interi
    vcvtdq2pd ymm3, xmm3             ; Converte in 4 doubles

    ; 4. CONFRONTO INDICI (Generazione Maschera)
    ; YMM4 sarà: Tutti 1 (NaN) se uguali, Tutti 0 se diversi
    vcmppd  ymm4, ymm1, ymm3, 0      ; _CMP_EQ_OQ
    
    ; 5. Carica e converti 4 elementi di W_SIGN
    vmovdqu xmm5, [rcx + r10*4]
    vcvtdq2pd ymm5, xmm5

    ; 6. CALCOLO DIFFERENZA SEGNI (Ottimizzazione XOR)
    ; Invece di fare (V - W) e poi Valore Assoluto, usiamo XOR.
    ; Se i segni sono 0.0 e 1.0:
    ; 0.0 ^ 0.0 = 0.0 (Segni uguali)
    ; 1.0 ^ 1.0 = 0.0 (Segni uguali) -> Nota: 1.0 è 0x3FF... i bit combaciano nel XOR
    ; 0.0 ^ 1.0 = 1.0 (Segni diversi)
    ; Questo risparmia istruzioni costose!
    vxorpd  ymm6, ymm2, ymm5         ; YMM6 = 0.0 (uguali) o 1.0 (diversi)

    ; 7. Formula finale: 1.0 - (2.0 * diff)
    vmulpd  ymm6, ymm6, ymm11        ; YMM6 = 0.0 o 2.0
    vsubpd  ymm7, ymm10, ymm6        ; YMM7 = 1.0 o -1.0

    ; 8. APPLICAZIONE MASCHERA
    ; Se l'indice non coincideva, YMM4 è 0, quindi il risultato diventa 0.0
    vandpd  ymm7, ymm7, ymm4         

    ; 9. ACCUMULO
    vaddpd  ymm0, ymm0, ymm7

    add     r10d, 4                  ; j += 4
    jmp     .inner_loop

.next_outer:
    inc     r9
    jmp     .outer_loop

.end_loops:
    ; --- RIDUZIONE FINALE (Horizontal Sum) ---
    ; Somma le 4 corsie di YMM0 in un unico scalare
    
    vextractf128 xmm1, ymm0, 1       ; Estrai parte alta (elementi 2,3)
    vaddpd  xmm0, xmm0, xmm1         ; Somma vettoriale: [0+2, 1+3]
    
    vpermilpd xmm1, xmm0, 1          ; Sposta elemento dispari in posizione pari
    vaddsd  xmm0, xmm0, xmm1         ; Somma scalare finale

    pop     rbp
    ret

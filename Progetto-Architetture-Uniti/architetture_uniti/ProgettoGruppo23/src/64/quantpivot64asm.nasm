; =============================================================
; FILE: quantpivot64asm.nasm
; ARCHITETTURA: x86-64 (Linux System V ABI)
; DESCRIZIONE: Implementazione ottimizzata AVX di KNN
; =============================================================

%include "sseutils64.nasm" 

default rel 

; =============================================================
; SEZIONE COSTANTI (Read-Only Data)
; =============================================================
section .rodata
    ; Maschera per valore assoluto: 0x7FFFF... mette a 0 il bit del segno
    ; C: fabs(val)
    abs_mask    dq 0x7FFFFFFFFFFFFFFF, 0
    
    ; Costante 1.0 per i calcoli di dist_approx
    one_val     dq 1.0
    
    ; Valore massimo Double (approx 1.79e308)
    ; C: FLT_MAX (o DBL_MAX)
    dbl_max     dq 0x7FEFFFFFFFFFFFFF, 0

    ; Stringhe per debug
    msg_nq      db  'nq:', 32, 0
    nl          db  10, 0

; =============================================================
; SEZIONE VARIABILI NON INIZIALIZZATE (.bss)
; =============================================================
section .bss
    alignb 16
    nq_val  resd    1

; =============================================================
; SEZIONE CODICE (.text)
; =============================================================
section .text

; Funzioni C esterne (Helper per memoria definiti in quantpivot64.c)
extern get_block
extern free_block

; -------------------------------------------------------------
; MACRO HELPER
; -------------------------------------------------------------
; C: malloc(size * count)
; Input: %1 = size, %2 = count
%macro  getmem  2
    mov     rdi, %1        
    mov     rsi, %2        
    call    get_block wrt ..plt
%endmacro

; C: free(ptr)
; Input: %1 = ptr
%macro  fremem  1
    mov     rdi, %1        
    call    free_block wrt ..plt
%endmacro


; =============================================================
; FUNZIONE: prova (Entry point test)
; =============================================================
global prova
prova:
    push    rbp
    mov     rbp, rsp
    leave
    ret


; =============================================================
; FUNZIONE: quantize_vector
; C: void quantize_vector(const type* vec, int D, int x, int* out_idx, int* out_sign)
; =============================================================
global quantize_vector
quantize_vector:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    ; --- Mappatura Argomenti ---
    ; RDI = vec
    ; RSI = D
    ; RDX = x
    ; RCX = out_idx
    ; R8  = out_sign

    mov     r12, rdi        ; r12 = vec
    movsxd  r13, esi        ; r13 = D
    movsxd  r14, edx        ; r14 = x
    mov     r15, rcx        ; r15 = out_idx
    mov     rbx, r8         ; rbx = out_sign

    vmovsd  xmm15, [rel abs_mask] ; Carica maschera assoluta per fabs()

    ; ---------------------------------------------------------
    ; C: for (int i = 0; i < x; i++) out_idx[i] = -1;
    ; ---------------------------------------------------------
    xor     rax, rax        ; i = 0
.init_loop:
    cmp     rax, r14
    jge     .init_done
    mov     dword [r15 + rax*4], -1
    inc     rax
    jmp     .init_loop
.init_done:

    ; ---------------------------------------------------------
    ; C: for (int i = 0; i < D; i++)
    ; ---------------------------------------------------------
    xor     r9, r9          ; r9 = i
.main_loop:
    cmp     r9, r13
    jge     .end_qv

    ; C: type val = vec[i]; type abs_val = fabs(val);
    vmovsd  xmm0, [r12 + r9*8]  ; val
    vandpd  xmm1, xmm0, xmm15   ; abs_val

    mov     rax, -1         ; pos = -1
    xor     r10, r10        ; j = 0

    ; ---------------------------------------------------------
    ; C: for(int j=0; j<x; j++) { if (out_idx[j] == -1 || abs_val > fabs(vec[out_idx[j]])) ... }
    ; ---------------------------------------------------------
.search_loop:
    cmp     r10, r14
    jge     .check_pos

    mov     r11d, [r15 + r10*4] ; Load out_idx[j]
    cmp     r11d, -1            ; if (out_idx[j] == -1)
    je      .found_spot

    movsxd  r11, r11d
    vmovsd  xmm2, [r12 + r11*8] ; Load vec[out_idx[j]]
    vandpd  xmm2, xmm2, xmm15   ; fabs(...)
    vucomisd xmm1, xmm2         ; compare abs_val > abs_old
    ja      .found_spot

    inc     r10
    jmp     .search_loop

.found_spot:
    mov     rax, r10        ; pos = j

.check_pos:
    cmp     rax, -1         ; if (pos != -1)
    je      .next_qv_iter

    ; ---------------------------------------------------------
    ; C: for (int k = x - 1; k > pos; k--) { shift right }
    ; ---------------------------------------------------------
    mov     r11, r14
    dec     r11             ; k = x - 1
.shift_loop:
    cmp     r11, rax
    jle     .insert
    ; out_idx[k] = out_idx[k-1]
    mov     r8d, [r15 + r11*4 - 4]
    mov     [r15 + r11*4], r8d     
    ; out_sign[k] = out_sign[k-1]
    mov     r8d, [rbx + r11*4 - 4]
    mov     [rbx + r11*4], r8d     
    dec     r11
    jmp     .shift_loop

.insert:
    ; ---------------------------------------------------------
    ; C: out_idx[pos] = i; out_sign[pos] = (val >= 0) ? 1 : -1;
    ; ---------------------------------------------------------
    mov     [r15 + rax*4], r9d      ; store index
    
    vxorpd  xmm2, xmm2, xmm2
    vucomisd xmm0, xmm2             ; compare val vs 0.0
    mov     r8d, -1
    mov     r11d, 1
    cmovae  r8d, r11d               ; if val >= 0 then 1 else -1
    mov     [rbx + rax*4], r8d      ; store sign

.next_qv_iter:
    inc     r9
    jmp     .main_loop

.end_qv:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    leave
    ret


; =============================================================
; FUNZIONE: dist_approx
; C: type dist_approx(int* v_idx, int* v_sign, int* w_idx, int* w_sign, int x)
; =============================================================
global dist_approx
dist_approx:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    mov     r12, rdi        ; v_idx
    mov     r13, rsi        ; v_sign
    mov     r14, rdx        ; w_idx
    mov     r15, rcx        ; w_sign
    movsxd  rbx, r8d        ; x

    vxorpd  xmm0, xmm0, xmm0    ; result = 0.0
    vmovsd  xmm1, [rel one_val] ; cost 1.0

    ; ---------------------------------------------------------
    ; C: Doppio loop for (i=0..x) { for (j=0..x) { ... } }
    ; ---------------------------------------------------------
    xor     r9, r9          ; i = 0
.loop_i:
    cmp     r9, rbx
    jge     .end_dist

    mov     eax, [r12 + r9*4]   ; v_idx[i]
    xor     r10, r10            ; j = 0
.loop_j:
    cmp     r10, rbx
    jge     .next_i_dist

    mov     r11d, [r14 + r10*4] ; w_idx[j]
    
    ; C: if (v_idx[i] == w_idx[j])
    cmp     eax, r11d
    jne     .next_j_dist        

    ; ---------------------------------------------------------
    ; OTTIMIZZAZIONE LOGICA RISPETTO AL C:
    ; Invece di 4 if/else if sui segni (v+, v-, w+, w-):
    ; Se (segno_v == segno_w) -> result += 1.0  (Equivale a v+w+ OR v-w-)
    ; Se (segno_v != segno_w) -> result -= 1.0  (Equivale a v+w- OR v-w+)
    ; ---------------------------------------------------------
    mov     r8d, [r13 + r9*4]   ; v_sign[i]
    mov     r11d, [r15 + r10*4] ; w_sign[j]

    cmp     r8d, r11d
    je      .signs_equal
.signs_diff:
    vsubsd  xmm0, xmm0, xmm1    ; result -= 1.0
    jmp     .next_j_dist
.signs_equal:
    vaddsd  xmm0, xmm0, xmm1    ; result += 1.0

.next_j_dist:
    inc     r10
    jmp     .loop_j
.next_i_dist:
    inc     r9
    jmp     .loop_i

.end_dist:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    leave
    ret


; =============================================================
; FUNZIONE: euclidean_distance
; C: type euclidean_distance(type* v, type* w, int D)
; =============================================================
global euclidean_distance
euclidean_distance:
    push    rbp
    mov     rbp, rsp

    vxorpd  xmm0, xmm0, xmm0    ; sum = 0.0
    xor     rax, rax            ; i = 0
    movsxd  rcx, edx            ; D

    test    rcx, rcx
    jle     .end_eucl

    ; ---------------------------------------------------------
    ; C: for (int i = 0; i < D; i++) sum += (v[i] - w[i])^2
    ; ---------------------------------------------------------
.loop_eucl:
    cmp     rax, rcx
    jge     .end_eucl

    vmovsd  xmm1, [rdi + rax*8]       ; v[i]
    vsubsd  xmm1, xmm1, [rsi + rax*8] ; diff = v[i] - w[i]
    vmulsd  xmm1, xmm1, xmm1          ; diff * diff
    vaddsd  xmm0, xmm0, xmm1          ; sum += ...

    inc     rax
    jmp     .loop_eucl

.end_eucl:
    vsqrtsd xmm0, xmm0, xmm0    ; return sqrt(sum)
    pop     rbp
    ret


; =============================================================
; FUNZIONE: fit
; C: void fit(params* input)
; =============================================================
global fit
fit:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    
    sub     rsp, 32 ; Spazio locale stack

    mov     rbx, rdi            ; input

    ; Carica parametri N, D, h, x dalla struct
    movsxd  r15, dword [rbx + 60] ; N
    movsxd  r14, dword [rbx + 64] ; D
    movsxd  r12, dword [rbx + 48] ; h
    movsxd  r13, dword [rbx + 56] ; x

    ; ---------------------------------------------------------
    ; 1. Allocazione Pivot P -> malloc(h * sizeof(int))
    ; ---------------------------------------------------------
    getmem  4, r12
    mov     [rbx + 8], rax      ; input->P

    ; C: step = N / h; for(j=0..h) P[j] = step * j
    mov     rax, r15
    xor     rdx, rdx
    div     r12
    mov     r8, rax             ; step

    mov     rcx, [rbx + 8]      ; P ptr
    xor     r9, r9              ; j=0
.loop_init_p:
    cmp     r9, r12
    jge     .done_init_p
    mov     rax, r8
    imul    rax, r9
    mov     [rcx + r9*4], eax   ; Store P[j]
    inc     r9
    jmp     .loop_init_p
.done_init_p:

    ; ---------------------------------------------------------
    ; 2. Allocazione Index -> malloc(N * h * sizeof(type))
    ; ---------------------------------------------------------
    mov     rax, r15
    imul    rax, r12
    getmem  8, rax
    mov     [rbx + 16], rax     ; input->index

    ; ---------------------------------------------------------
    ; 3. Allocazione Buffer Temporanei
    ; ---------------------------------------------------------
    mov     rax, r12
    imul    rax, r13            ; h * x

    push    rax                 ; Save size
    getmem  4, rax
    mov     [rbp - 8], rax      ; pivot_idxs
    pop     rax

    push    rax
    getmem  4, rax
    mov     [rbp - 16], rax     ; pivot_signs
    pop     rax

    getmem  4, r13
    mov     [rbp - 24], rax     ; point_idx
    getmem  4, r13
    mov     [rbp - 32], rax     ; point_sign

    ; ---------------------------------------------------------
    ; Pre-quantizzazione Pivot: for(j=0..h) quantize(DS[P[j]])
    ; ---------------------------------------------------------
    xor     r9, r9
.loop_prequant:
    cmp     r9, r12
    jge     .done_prequant

    mov     rcx, [rbx + 8]      ; P
    mov     eax, [rcx + r9*4]   ; P[j]
    cdqe
    imul    rax, r14            ; * D
    shl     rax, 3              ; * 8
    add     rax, [rbx]          ; DS base
    mov     rdi, rax            ; vec = &DS[P[j]*D]

    mov     rsi, r14            ; D
    mov     rdx, r13            ; x
    
    ; Calcolo indirizzi buffer pivot_idxs[j*x]
    mov     rax, r9
    imul    rax, r13
    shl     rax, 2
    mov     rcx, [rbp - 8]
    add     rcx, rax            ; &pivot_idxs
    mov     r8, [rbp - 16]
    add     r8, rax             ; &pivot_signs

    push    r9
    call    quantize_vector
    pop     r9

    inc     r9
    jmp     .loop_prequant
.done_prequant:

    ; ---------------------------------------------------------
    ; 4. Costruzione Indice: for(i=0..N) { quantize(pt); for(j=0..h) dist(); }
    ; ---------------------------------------------------------
    xor     r9, r9              ; i = 0
.loop_outer_i:
    cmp     r9, r15
    jge     .cleanup_fit

    ; Quantize punto corrente i
    mov     rax, r9
    imul    rax, r14
    shl     rax, 3
    add     rax, [rbx]          ; vec = &DS[i*D]
    mov     rdi, rax

    mov     rsi, r14
    mov     rdx, r13
    mov     rcx, [rbp - 24]
    mov     r8,  [rbp - 32]

    push    r9
    call    quantize_vector
    pop     r9

    ; Calcola distanze con tutti i pivot
    xor     r10, r10            ; j = 0
.loop_inner_j:
    cmp     r10, r12
    jge     .next_i_fit

    mov     rdi, [rbp - 24]     ; point_idx
    mov     rsi, [rbp - 32]     ; point_sign

    ; Indirizzo pivot j
    mov     rax, r10
    imul    rax, r13
    shl     rax, 2
    mov     rdx, [rbp - 8]
    add     rdx, rax            ; pivot_idxs[j]
    mov     rcx, [rbp - 16]
    add     rcx, rax            ; pivot_signs[j]
    mov     r8, r13

    push    r9
    push    r10
    call    dist_approx
    pop     r10
    pop     r9

    ; Store in index[i*h + j]
    mov     rax, r9
    imul    rax, r12
    add     rax, r10
    shl     rax, 3
    mov     rcx, [rbx + 16]
    vmovsd  [rcx + rax], xmm0

    inc     r10
    jmp     .loop_inner_j

.next_i_fit:
    inc     r9
    jmp     .loop_outer_i

.cleanup_fit:
    fremem  [rbp - 32]
    fremem  [rbp - 24]
    fremem  [rbp - 16]
    fremem  [rbp - 8]

    add     rsp, 32
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret


; =============================================================
; FUNZIONE: predict
; C: void predict(params* input)
; =============================================================
global predict
predict:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    
    sub     rsp, 96

    mov     rbx, rdi            ; input
    movsxd  r15, dword [rbx + 60] ; N
    movsxd  r12, dword [rbx + 48] ; h
    movsxd  r14, dword [rbx + 52] ; k
    movsxd  r13, dword [rbx + 56] ; x
    
    mov     eax, [rbx + 64]     ; D
    mov     [rbp - 88], eax     ; Salva D locale per liberare registri

    ; ---------------------------------------------------------
    ; 1. Allocazione buffer temporanei (q_idx, p_idx, dists...)
    ; ---------------------------------------------------------
    getmem  4, r13
    mov     [rbp - 8], rax      ; q_idx
    getmem  4, r13
    mov     [rbp - 16], rax     ; q_sign
    getmem  4, r13
    mov     [rbp - 24], rax     ; p_idx
    getmem  4, r13
    mov     [rbp - 32], rax     ; p_sign
    getmem  4, r13
    mov     [rbp - 40], rax     ; v_idx
    getmem  4, r13
    mov     [rbp - 48], rax     ; v_sign
    getmem  8, r12
    mov     [rbp - 56], rax     ; dist_query_pivots
    getmem  4, r14
    mov     [rbp - 64], rax     ; knn_ids
    getmem  8, r14
    mov     [rbp - 72], rax     ; knn_dists

    ; ---------------------------------------------------------
    ; Loop Queries: for (int iq = 0; iq < nq; iq++)
    ; ---------------------------------------------------------
    mov     dword [rbp - 80], 0 ; iq = 0
.loop_iq:
    mov     eax, [rbp - 80]
    cmp     eax, [rbx + 68]     ; nq
    jge     .cleanup_predict

    ; C: Init KNN list {-1, FLT_MAX}
    xor     r9, r9
    vmovsd  xmm0, [rel dbl_max] ; Load Double MAX
    mov     rdi, [rbp - 64]     ; ids
    mov     rsi, [rbp - 72]     ; dists
.loop_init_knn:
    cmp     r9, r14
    jge     .done_init_knn
    mov     dword [rdi + r9*4], -1
    vmovsd  [rsi + r9*8], xmm0
    inc     r9
    jmp     .loop_init_knn
.done_init_knn:

    ; C: quantize_vector(query_vec)
    mov     eax, [rbp - 80]
    movsxd  rcx, dword [rbp - 88] ; D
    imul    rax, rcx
    shl     rax, 3
    add     rax, [rbx + 24]     ; Q base
    mov     rdi, rax

    mov     rsi, rcx
    mov     rdx, r13
    mov     rcx, [rbp - 8]
    mov     r8,  [rbp - 16]
    call    quantize_vector

    ; ---------------------------------------------------------
    ; 2. Calc dist Query <-> Pivots
    ; ---------------------------------------------------------
    xor     r9, r9
.loop_pivots:
    cmp     r9, r12
    jge     .done_pivots

    ; Recupera pivot e quantizza
    mov     rcx, [rbx + 8]      ; P
    mov     eax, [rcx + r9*4]
    cdqe
    movsxd  rcx, dword [rbp - 88] ; D
    imul    rax, rcx
    shl     rax, 3
    add     rax, [rbx]          ; DS
    mov     rdi, rax

    mov     rsi, rcx
    mov     rdx, r13
    mov     rcx, [rbp - 24]
    mov     r8,  [rbp - 32]
    push    r9
    call    quantize_vector
    pop     r9

    ; Calcola distanza approx
    mov     rdi, [rbp - 8]
    mov     rsi, [rbp - 16]
    mov     rdx, [rbp - 24]
    mov     rcx, [rbp - 32]
    mov     r8, r13
    push    r9
    call    dist_approx
    pop     r9

    mov     rax, [rbp - 56]
    vmovsd  [rax + r9*8], xmm0  ; Store in dist_query_pivots
    inc     r9
    jmp     .loop_pivots
.done_pivots:

    ; ---------------------------------------------------------
    ; 3. Scan Dataset: for (v = 0; v < N; v++)
    ; ---------------------------------------------------------
    xor     r10, r10            ; v=0
.loop_scan_v:
    cmp     r10, r15
    jge     .done_scan_v

    ; C: Find d_k_max in KNN list
    mov     rax, -1             ; max_pos
    mov     rcx, -1
    cvtsi2sd xmm1, rcx          ; d_k_max = -1.0
    xor     r9, r9
    mov     rdx, [rbp - 72]
.loop_find_max:
    cmp     r9, r14
    jge     .done_find_max
    vmovsd  xmm2, [rdx + r9*8]
    vucomisd xmm2, xmm1
    jbe     .skip_max
    vmovsd  xmm1, xmm2
    mov     rax, r9
.skip_max:
    inc     r9
    jmp     .loop_find_max
.done_find_max:

    ; ---------------------------------------------------------
    ; 4. Lower Bound Check (d_pvt_star)
    ; ---------------------------------------------------------
    vxorpd  xmm2, xmm2, xmm2    ; d_pvt_star = 0.0
    vmovsd  xmm15, [rel abs_mask]
    xor     r9, r9
    mov     rcx, r10
    imul    rcx, r12
    shl     rcx, 3
    add     rcx, [rbx + 16]     ; index[v]
    mov     rdx, [rbp - 56]     ; pivot dists
.loop_lb:
    cmp     r9, r12
    jge     .done_lb
    vmovsd  xmm3, [rcx + r9*8]
    vsubsd  xmm3, xmm3, [rdx + r9*8]
    vandpd  xmm3, xmm3, xmm15   ; fabs()
    vucomisd xmm3, xmm2
    jbe     .skip_lb
    vmovsd  xmm2, xmm3          ; Update max diff
.skip_lb:
    inc     r9
    jmp     .loop_lb
.done_lb:

    ; C: if (d_pvt_star < d_k_max)
    vucomisd xmm2, xmm1
    jae     .next_v

    ; ---------------------------------------------------------
    ; 5. Calcolo Distanza Approx Reale: quantize(v) + dist()
    ; ---------------------------------------------------------
    mov     rcx, r10
    movsxd  rsi, dword [rbp - 88]
    imul    rcx, rsi
    shl     rcx, 3
    add     rcx, [rbx]
    mov     rdi, rcx
    
    mov     rdx, r13
    mov     rcx, [rbp - 40]
    mov     r8,  [rbp - 48]
    
    push    r10
    push    rax
    sub     rsp, 16
    vmovsd  [rsp], xmm1         ; Salva d_k_max
    call    quantize_vector

    mov     rdi, [rbp - 8]
    mov     rsi, [rbp - 16]
    mov     rdx, [rbp - 40]
    mov     rcx, [rbp - 48]
    mov     r8, r13
    call    dist_approx
    
    vmovsd  xmm1, [rsp]         ; Recupera d_k_max
    add     rsp, 16
    pop     rax
    pop     r10

    ; C: if (dist < d_k_max) -> Update KNN
    vucomisd xmm0, xmm1
    jae     .next_v

    mov     rcx, [rbp - 64]
    mov     [rcx + rax*4], r10d ; Update ID
    mov     rcx, [rbp - 72]
    vmovsd  [rcx + rax*8], xmm0 ; Update Dist

.next_v:
    inc     r10
    jmp     .loop_scan_v
.done_scan_v:

    ; ---------------------------------------------------------
    ; 7. Raffinamento: Calcolo Euclideo finale sui K candidati
    ; ---------------------------------------------------------
    xor     r9, r9
.loop_refine:
    cmp     r9, r14
    jge     .next_query

    mov     rcx, [rbp - 64]
    mov     eax, [rcx + r9*4]
    cmp     eax, -1
    je      .inf_dist

    ; Calcola Distanza Vera
    mov     r11d, [rbp - 80]
    movsxd  rsi, dword [rbp - 88]
    imul    r11, rsi
    shl     r11, 3
    add     r11, [rbx + 24]     ; Q ptr
    mov     rdi, r11

    movsxd  rcx, eax
    imul    rcx, rsi
    shl     rcx, 3
    add     rcx, [rbx]          ; DS ptr
    mov     rsi, rcx
    mov     rdx, [rbp - 88]

    push    r9
    push    rax
    call    euclidean_distance
    pop     rax
    pop     r9
    jmp     .store_res

.inf_dist:
    vmovsd  xmm0, [rel dbl_max]

.store_res:
    ; Scrivi risultato finale
    mov     r10d, [rbp - 80]
    imul    r10, r14
    add     r10, r9
    
    mov     rcx, [rbx + 32]
    mov     [rcx + r10*4], eax
    mov     rcx, [rbx + 40]
    vmovsd  [rcx + r10*8], xmm0

    inc     r9
    jmp     .loop_refine

.next_query:
    inc     dword [rbp - 80]
    jmp     .loop_iq

.cleanup_predict:
    ; Free di tutti i buffer
    fremem  [rbp - 8]
    fremem  [rbp - 16]
    fremem  [rbp - 24]
    fremem  [rbp - 32]
    fremem  [rbp - 40]
    fremem  [rbp - 48]
    fremem  [rbp - 56]
    fremem  [rbp - 64]
    fremem  [rbp - 72]

    add     rsp, 96
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret
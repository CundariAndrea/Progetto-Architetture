; -----------------------------------------------------------------------------
; Funzione quantize_vector ottimizzata con AVX (YMM 256-bit)
; Versione "No-Malloc": Il buffer temporaneo viene passato dal C.
;
; Firma C aggiornata:
; void quantize_vector(const double *vec, const int D, const int x, 
;                      int *out_idx, int *out_sign, double *shadow_vals);
;
; Parametri (System V ABI):
; RDI = vec
; ESI = D
; EDX = x
; RCX = out_idx
; R8  = out_sign
; R9  = shadow_vals (NUOVO: Buffer pre-allocato di dimensione x * sizeof(double))
; -----------------------------------------------------------------------------

global quantize_vector

section .data
    align 32
    abs_mask: dq 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF
    dbl_max:  dq 1.7976931348623157e+308

    default rel

section .text

quantize_vector:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    
    ; Non serve più allocare spazio sullo stack per chiamate a funzione esterne

    ; --- Mapping Registri ---
    mov     r12, rdi            ; r12 = vec
    mov     r13d, esi           ; r13d = D
    mov     r14d, edx           ; r14d = x
    mov     r15, rcx            ; r15 = out_idx
    mov     rbx, r8             ; rbx = out_sign
    ; R9 contiene già shadow_vals, lo usiamo direttamente.

    ; ---------------------------------------------------------
    ; FASE 1: Riempimento Iniziale (Primi x elementi)
    ; ---------------------------------------------------------
    movsd   xmm0, [dbl_max]     ; min_val = MAX
    xor     rax, rax            ; min_pos = 0
    xor     rsi, rsi            ; i = 0

    vmovapd ymm15, [abs_mask]   ; Carica maschera ABS

.init_loop:
    cmp     esi, r14d           ; i < x
    jge     .init_done

    ; Carica vec[i]
    movsd   xmm1, [r12 + rsi*8]
    
    ; out_idx[i] = i
    mov     dword [r15 + rsi*4], esi
    
    ; out_sign[i]
    xor     edx, edx
    xorpd   xmm2, xmm2
    ucomisd xmm1, xmm2
    setae   dl
    mov     dword [rbx + rsi*4], edx

    ; shadow_vals[i] = abs(val) usando R9
    vpand   xmm1, xmm1, xmm15
    movsd   [r9 + rsi*8], xmm1  ; Scrive nel buffer passato dal C

    ; Update min
    ucomisd xmm1, xmm0
    jae     .skip_min_update
    movapd  xmm0, xmm1
    mov     rax, rsi
.skip_min_update:
    inc     rsi
    jmp     .init_loop

.init_done:
    vbroadcastsd ymm0, xmm0     ; Broadcast min_val

    ; ---------------------------------------------------------
    ; FASE 2: Loop Principale AVX (Scan & Swap)
    ; ---------------------------------------------------------
    mov     rcx, rsi            ; i = x
    mov     rdx, r13
    sub     rdx, 4              ; Limite AVX (D-4)

.main_loop:
    cmp     rcx, rdx
    jg      .scalar_cleanup

    ; --- FAST PATH ---
    vmovupd ymm1, [r12 + rcx*8] ; Load 4 doubles
    vandpd  ymm2, ymm1, ymm15   ; ABS

    ; Confronta tutti e 4 col minimo
    vcmppd  ymm3, ymm2, ymm0, 14 ; _CMP_GT_OS (14)
    
    vmovmskpd edi, ymm3
    test    edi, edi
    jnz     .process_batch

    ; Nessuno è maggiore del minimo -> Salta
    add     rcx, 4
    jmp     .main_loop

.process_batch:
    ; --- SLOW PATH (Processa i 4 elementi uno a uno) ---
    push    rcx
    mov     r8d, 4

.batch_inner:
    movsd   xmm1, [r12 + rcx*8]
    vpand   xmm2, xmm1, xmm15   ; abs
    
    ucomisd xmm2, xmm0          ; cmp vs min (parte bassa YMM0)
    jbe     .next_in_batch

    ; --- Update (Trovato nuovo massimo locale) ---
    mov     dword [r15 + rax*4], ecx    ; out_idx
    
    xor     edi, edi
    xorpd   xmm3, xmm3
    ucomisd xmm1, xmm3
    setae   dil
    mov     dword [rbx + rax*4], edi    ; out_sign

    movsd   [r9 + rax*8], xmm2          ; shadow_vals

    ; Ricalcola minimo (Linear Scan su shadow_vals in R9)
    movsd   xmm0, [dbl_max]
    xor     rax, rax
    xor     rsi, rsi
.find_min_loop:
    cmp     esi, r14d
    jge     .min_found
    movsd   xmm3, [r9 + rsi*8]
    ucomisd xmm3, xmm0
    jae     .next_k
    movapd  xmm0, xmm3
    mov     rax, rsi
.next_k:
    inc     rsi
    jmp     .find_min_loop
.min_found:
    vbroadcastsd ymm0, xmm0     ; Aggiorna broadcast

.next_in_batch:
    inc     rcx
    dec     r8d
    jnz     .batch_inner

    pop     rcx
    add     rcx, 4
    jmp     .main_loop

.scalar_cleanup:
    ; Gestione coda (ultimi <4 elementi)
    cmp     rcx, r13
    jge     .sort_phase

    movsd   xmm1, [r12 + rcx*8]
    ; CORREZIONE QUI: Usa xmm15 (128-bit) invece di ymm15 (256-bit) per operazione scalare
    vpand   xmm2, xmm1, xmm15   ; abs (usiamo XMM15 basso)
    
    ucomisd xmm2, xmm0
    jbe     .scalar_next

    ; Update scalare
    mov     dword [r15 + rax*4], ecx
    xor     edi, edi
    xorpd   xmm3, xmm3
    ucomisd xmm1, xmm3
    setae   dil
    mov     dword [rbx + rax*4], edi
    movsd   [r9 + rax*8], xmm2

    ; Ricalcola min
    movsd   xmm0, [dbl_max]
    xor     rax, rax
    xor     rsi, rsi
.find_min_loop_tail:
    cmp     esi, r14d
    jge     .min_found_tail
    movsd   xmm3, [r9 + rsi*8]
    ucomisd xmm3, xmm0
    jae     .next_k_tail
    movapd  xmm0, xmm3
    mov     rax, rsi
.next_k_tail:
    inc     rsi
    jmp     .find_min_loop_tail
.min_found_tail:
    ; No broadcast needed

.scalar_next:
    inc     rcx
    jmp     .scalar_cleanup

    ; ---------------------------------------------------------
    ; FASE 3: Insertion Sort
    ; ---------------------------------------------------------
.sort_phase:
    ; NON chiamiamo free(R9). La memoria è gestita dal C.

    mov     rcx, 1              ; i = 1
.sort_outer:
    cmp     ecx, r14d
    jge     .done

    mov     r8d, dword [r15 + rcx*4] ; key_idx
    mov     r10d, dword [rbx + rcx*4]; key_sign (usiamo R10d invece di R9d che è il puntatore buffer!)
    
    mov     rsi, rcx
    dec     rsi                 ; j = i - 1

.sort_inner:
    cmp     rsi, 0
    jl      .sort_insert

    mov     edi, dword [r15 + rsi*4]
    cmp     edi, r8d
    jle     .sort_insert

    mov     dword [r15 + rsi*4 + 4], edi
    mov     eax, dword [rbx + rsi*4]
    mov     dword [rbx + rsi*4 + 4], eax

    dec     rsi
    jmp     .sort_inner

.sort_insert:
    mov     dword [r15 + rsi*4 + 4], r8d
    mov     dword [rbx + rsi*4 + 4], r10d

    inc     rcx
    jmp     .sort_outer

.done:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

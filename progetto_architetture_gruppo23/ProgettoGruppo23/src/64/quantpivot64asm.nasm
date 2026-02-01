; =============================================================
; FILE: quantpivot64asm.nasm
; ARCHITETTURA: x86-64
; DESCRIZIONE: Kernel computazionali ottimizzati AVX
; =============================================================

%include "sseutils64.nasm" 
default rel 

section .rodata
    abs_mask    dq 0x7FFFFFFFFFFFFFFF, 0
    one_val     dq 1.0

section .text

; -------------------------------------------------------------
; quantize_vector
; -------------------------------------------------------------
global quantize_vector
quantize_vector:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    mov     r12, rdi
    movsxd  r13, esi
    movsxd  r14, edx
    mov     r15, rcx
    mov     rbx, r8

    vmovsd  xmm15, [rel abs_mask]

    xor     rax, rax
.init_loop:
    cmp     rax, r14
    jge     .init_done
    mov     dword [r15 + rax*4], -1
    inc     rax
    jmp     .init_loop
.init_done:

    xor     r9, r9
.main_loop:
    cmp     r9, r13
    jge     .end_qv

    vmovsd  xmm0, [r12 + r9*8]
    vandpd  xmm1, xmm0, xmm15
    vmovq   rdx, xmm1

    mov     rax, -1
    xor     r10, r10

.search_loop:
    cmp     r10, r14
    jge     .check_pos

    mov     r11d, [r15 + r10*4]
    cmp     r11d, -1
    je      .found_spot

    movsxd  r11, r11d
    vmovsd  xmm2, [r12 + r11*8]
    vandpd  xmm2, xmm2, xmm15
    vmovq   rcx, xmm2
    
    cmp     rdx, rcx
    ja      .found_spot

    inc     r10
    jmp     .search_loop

.found_spot:
    mov     rax, r10

.check_pos:
    cmp     rax, -1
    je      .next_qv_iter

    mov     r11, r14
    dec     r11
.shift_loop:
    cmp     r11, rax
    jle     .insert
    mov     r8d, [r15 + r11*4 - 4]
    mov     [r15 + r11*4], r8d      
    mov     r8d, [rbx + r11*4 - 4]
    mov     [rbx + r11*4], r8d      
    dec     r11
    jmp     .shift_loop

.insert:
    mov     [r15 + rax*4], r9d
    
    vmovq   rcx, xmm0
    test    rcx, rcx
    
    mov     r8d, -1
    mov     r11d, 1
    cmovns  r8d, r11d
    
    mov     [rbx + rax*4], r8d

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


; -------------------------------------------------------------
; dist_approx
; -------------------------------------------------------------
global dist_approx
dist_approx:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    mov     r12, rdi
    mov     r13, rsi
    mov     r14, rdx
    mov     r15, rcx
    movsxd  rbx, r8d

    vxorpd  xmm0, xmm0, xmm0
    vmovsd  xmm1, [rel one_val]

    xor     r9, r9
.loop_i:
    cmp     r9, rbx
    jge     .end_dist

    mov     eax, [r12 + r9*4]
    xor     r10, r10
.loop_j:
    cmp     r10, rbx
    jge     .next_i_dist

    mov     r11d, [r14 + r10*4]
    cmp     eax, r11d
    jne     .next_j_dist        

    mov     r8d, [r13 + r9*4]
    mov     r11d, [r15 + r10*4]

    cmp     r8d, r11d
    je      .signs_equal
.signs_diff:
    vsubsd  xmm0, xmm0, xmm1
    jmp     .next_j_dist
.signs_equal:
    vaddsd  xmm0, xmm0, xmm1

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


; -------------------------------------------------------------
; euclidean_distance
; -------------------------------------------------------------
global euclidean_distance
euclidean_distance:
    push    rbp
    mov     rbp, rsp

    vxorpd  xmm0, xmm0, xmm0
    xor     rax, rax
    movsxd  rcx, edx

    test    rcx, rcx
    jle     .end_eucl

.loop_eucl:
    cmp     rax, rcx
    jge     .end_eucl

    vmovsd  xmm1, [rdi + rax*8]
    vsubsd  xmm1, xmm1, [rsi + rax*8]
    vmulsd  xmm1, xmm1, xmm1
    vaddsd  xmm0, xmm0, xmm1

    inc     rax
    jmp     .loop_eucl

.end_eucl:
    vsqrtsd xmm0, xmm0, xmm0
    pop     rbp
    ret
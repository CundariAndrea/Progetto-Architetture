; void abs_sse32(float* out_abs, const float* in, int D);
; rdi=out_abs, rsi=in, edx=D
global abs_sse32
section .text

abs_sse32:
    push rbp
    mov rbp, rsp

    ; mask = 0x7fffffff in tutti i lane (toglie bit segno)
    mov eax, 0x7fffffff
    movd xmm7, eax
    shufps xmm7, xmm7, 0          ; xmm7 = [mask mask mask mask]

    xor ecx, ecx                  ; i = 0
.loop:
    cmp ecx, edx
    jge .done

    movups xmm0, [rsi + rcx*4]    ; carica 4 float
    andps xmm0, xmm7              ; abs
    movups [rdi + rcx*4], xmm0    ; salva 4 abs

    add ecx, 4
    jmp .loop

.done:
    mov rsp, rbp
    pop rbp
    ret

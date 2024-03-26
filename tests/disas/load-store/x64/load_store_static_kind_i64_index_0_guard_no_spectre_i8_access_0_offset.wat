;;! target = "x86_64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation=false -W memory64 -O static-memory-forced -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i64 1)

  (func (export "do_store") (param i64 i32)
    local.get 0
    local.get 1
    i32.store8 offset=0)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load8_u offset=0))

;; wasm[0]::function[0]:
;;    0: pushq   %rbp
;;    1: movq    %rsp, %rbp
;;    4: cmpq    0x15(%rip), %rdx
;;    b: ja      0x1e
;;   11: movq    0x50(%rdi), %r10
;;   15: movb    %cl, (%r10, %rdx)
;;   19: movq    %rbp, %rsp
;;   1c: popq    %rbp
;;   1d: retq
;;   1e: ud2
;;
;; wasm[0]::function[1]:
;;   30: pushq   %rbp
;;   31: movq    %rsp, %rbp
;;   34: cmpq    0x1d(%rip), %rdx
;;   3b: ja      0x4f
;;   41: movq    0x50(%rdi), %r10
;;   45: movzbq  (%r10, %rdx), %rax
;;   4a: movq    %rbp, %rsp
;;   4d: popq    %rbp
;;   4e: retq
;;   4f: ud2
;;   51: addb    %al, (%rax)
;;   53: addb    %al, (%rax)
;;   55: addb    %al, (%rax)
;;   57: addb    %bh, %bh
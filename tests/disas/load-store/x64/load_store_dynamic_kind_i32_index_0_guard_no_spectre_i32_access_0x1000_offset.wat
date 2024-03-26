;;! target = "x86_64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation=false -O static-memory-maximum-size=0 -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i32 1)

  (func (export "do_store") (param i32 i32)
    local.get 0
    local.get 1
    i32.store offset=0x1000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load offset=0x1000))

;; wasm[0]::function[0]:
;;    0: pushq   %rbp
;;    1: movq    %rsp, %rbp
;;    4: movq    0x58(%rdi), %r9
;;    8: movl    %edx, %r11d
;;    b: subq    $0x1004, %r9
;;   12: cmpq    %r9, %r11
;;   15: ja      0x2c
;;   1b: movq    0x50(%rdi), %rdi
;;   1f: movl    %ecx, 0x1000(%rdi, %r11)
;;   27: movq    %rbp, %rsp
;;   2a: popq    %rbp
;;   2b: retq
;;   2c: ud2
;;
;; wasm[0]::function[1]:
;;   30: pushq   %rbp
;;   31: movq    %rsp, %rbp
;;   34: movq    0x58(%rdi), %r9
;;   38: movl    %edx, %r11d
;;   3b: subq    $0x1004, %r9
;;   42: cmpq    %r9, %r11
;;   45: ja      0x5c
;;   4b: movq    0x50(%rdi), %rdi
;;   4f: movl    0x1000(%rdi, %r11), %eax
;;   57: movq    %rbp, %rsp
;;   5a: popq    %rbp
;;   5b: retq
;;   5c: ud2
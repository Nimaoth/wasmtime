;;! target = "aarch64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation=false -W memory64 -O static-memory-maximum-size=0 -O static-memory-guard-size=4294967295 -O dynamic-memory-guard-size=4294967295"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i64 1)

  (func (export "do_store") (param i64 i32)
    local.get 0
    local.get 1
    i32.store8 offset=0x1000)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load8_u offset=0x1000))

;; wasm[0]::function[0]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       ldr     x8, [x2, #0x68]
;;       cmp     x4, x8
;;       cset    x9, hi
;;       uxtb    w9, w9
;;       cbnz    x9, #0x30
;;   1c: ldr     x10, [x2, #0x60]
;;       add     x10, x10, #1, lsl #12
;;       strb    w5, [x10, x4]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
;;   30: .byte   0x1f, 0xc1, 0x00, 0x00
;;
;; wasm[0]::function[1]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       ldr     x8, [x2, #0x68]
;;       cmp     x4, x8
;;       cset    x9, hi
;;       uxtb    w9, w9
;;       cbnz    x9, #0x70
;;   5c: ldr     x10, [x2, #0x60]
;;       add     x9, x10, #1, lsl #12
;;       ldrb    w2, [x9, x4]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
;;   70: .byte   0x1f, 0xc1, 0x00, 0x00

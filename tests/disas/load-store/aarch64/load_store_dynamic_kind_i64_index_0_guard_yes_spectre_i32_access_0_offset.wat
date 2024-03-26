;;! target = "aarch64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -W memory64 -O static-memory-maximum-size=0 -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i64 1)

  (func (export "do_store") (param i64 i32)
    local.get 0
    local.get 1
    i32.store offset=0)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load offset=0))

;; wasm[0]::function[0]:
;;    0: stp     x29, x30, [sp, #-0x10]!
;;    4: mov     x29, sp
;;    8: ldr     x11, [x0, #0x58]
;;    c: ldr     x10, [x0, #0x50]
;;   10: sub     x11, x11, #4
;;   14: mov     x12, #0
;;   18: add     x10, x10, x2
;;   1c: cmp     x2, x11
;;   20: csel    x11, x12, x10, hi
;;   24: csdb
;;   28: str     w3, [x11]
;;   2c: ldp     x29, x30, [sp], #0x10
;;   30: ret
;;
;; wasm[0]::function[1]:
;;   40: stp     x29, x30, [sp, #-0x10]!
;;   44: mov     x29, sp
;;   48: ldr     x11, [x0, #0x58]
;;   4c: ldr     x10, [x0, #0x50]
;;   50: sub     x11, x11, #4
;;   54: mov     x12, #0
;;   58: add     x10, x10, x2
;;   5c: cmp     x2, x11
;;   60: csel    x11, x12, x10, hi
;;   64: csdb
;;   68: ldr     w0, [x11]
;;   6c: ldp     x29, x30, [sp], #0x10
;;   70: ret
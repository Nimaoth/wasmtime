;;! target = "s390x"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation=false -O static-memory-forced -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i32 1)

  (func (export "do_store") (param i32 i32)
    local.get 0
    local.get 1
    i32.store8 offset=0)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load8_u offset=0))

;; wasm[0]::function[0]:
;;    0: stmg    %r14, %r15, 0x70(%r15)
;;    6: lgr     %r1, %r15
;;    a: aghi    %r15, -0xa0
;;    e: stg     %r1, 0(%r15)
;;   14: lgr     %r3, %r4
;;   18: lg      %r4, 0x50(%r2)
;;   1e: lgr     %r2, %r3
;;   22: llgfr   %r2, %r2
;;   26: stc     %r5, 0(%r2, %r4)
;;   2a: lmg     %r14, %r15, 0x110(%r15)
;;   30: br      %r14
;;
;; wasm[0]::function[1]:
;;   34: stmg    %r14, %r15, 0x70(%r15)
;;   3a: lgr     %r1, %r15
;;   3e: aghi    %r15, -0xa0
;;   42: stg     %r1, 0(%r15)
;;   48: lgr     %r5, %r4
;;   4c: lg      %r4, 0x50(%r2)
;;   52: lgr     %r2, %r5
;;   56: llgfr   %r5, %r2
;;   5a: llc     %r2, 0(%r5, %r4)
;;   60: lmg     %r14, %r15, 0x110(%r15)
;;   66: br      %r14
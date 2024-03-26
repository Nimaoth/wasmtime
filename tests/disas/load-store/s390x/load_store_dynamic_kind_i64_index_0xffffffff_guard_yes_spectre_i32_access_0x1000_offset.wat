;;! target = "s390x"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -W memory64 -O static-memory-maximum-size=0 -O static-memory-guard-size=4294967295 -O dynamic-memory-guard-size=4294967295"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i64 1)

  (func (export "do_store") (param i64 i32)
    local.get 0
    local.get 1
    i32.store offset=0x1000)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load offset=0x1000))

;; wasm[0]::function[0]:
;;    0: stmg    %r6, %r15, 0x30(%r15)
;;    6: lgr     %r1, %r15
;;    a: aghi    %r15, -0xa0
;;    e: stg     %r1, 0(%r15)
;;   14: lg      %r6, 0x58(%r2)
;;   1a: lghi    %r3, 0
;;   1e: lgr     %r7, %r4
;;   22: ag      %r7, 0x50(%r2)
;;   28: aghik   %r2, %r7, 0x1000
;;   2e: clgr    %r4, %r6
;;   32: locgrh  %r2, %r3
;;   36: strv    %r5, 0(%r2)
;;   3c: lmg     %r6, %r15, 0xd0(%r15)
;;   42: br      %r14
;;
;; wasm[0]::function[1]:
;;   44: stmg    %r6, %r15, 0x30(%r15)
;;   4a: lgr     %r1, %r15
;;   4e: aghi    %r15, -0xa0
;;   52: stg     %r1, 0(%r15)
;;   58: lg      %r5, 0x58(%r2)
;;   5e: lghi    %r3, 0
;;   62: lgr     %r6, %r4
;;   66: ag      %r6, 0x50(%r2)
;;   6c: aghik   %r2, %r6, 0x1000
;;   72: clgr    %r4, %r5
;;   76: locgrh  %r2, %r3
;;   7a: lrv     %r2, 0(%r2)
;;   80: lmg     %r6, %r15, 0xd0(%r15)
;;   86: br      %r14
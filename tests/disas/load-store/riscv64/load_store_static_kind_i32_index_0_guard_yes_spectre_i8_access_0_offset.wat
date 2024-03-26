;;! target = "riscv64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -O static-memory-forced -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

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
;;    0: addi    sp, sp, -0x10
;;    4: sd      ra, 8(sp)
;;    8: sd      s0, 0(sp)
;;    c: mv      s0, sp
;;   10: ld      a4, 0x50(a0)
;;   14: slli    a1, a2, 0x20
;;   18: srli    a5, a1, 0x20
;;   1c: add     a2, a4, a5
;;   20: sb      a3, 0(a2)
;;   24: ld      ra, 8(sp)
;;   28: ld      s0, 0(sp)
;;   2c: addi    sp, sp, 0x10
;;   30: ret
;;
;; wasm[0]::function[1]:
;;   34: addi    sp, sp, -0x10
;;   38: sd      ra, 8(sp)
;;   3c: sd      s0, 0(sp)
;;   40: mv      s0, sp
;;   44: ld      a3, 0x50(a0)
;;   48: slli    a1, a2, 0x20
;;   4c: srli    a4, a1, 0x20
;;   50: add     a2, a3, a4
;;   54: lbu     a0, 0(a2)
;;   58: ld      ra, 8(sp)
;;   5c: ld      s0, 0(sp)
;;   60: addi    sp, sp, 0x10
;;   64: ret
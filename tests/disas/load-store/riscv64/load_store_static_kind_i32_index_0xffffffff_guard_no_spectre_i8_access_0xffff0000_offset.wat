;;! target = "riscv64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation=false -O static-memory-forced -O static-memory-guard-size=4294967295 -O dynamic-memory-guard-size=4294967295"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i32 1)

  (func (export "do_store") (param i32 i32)
    local.get 0
    local.get 1
    i32.store8 offset=0xffff0000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load8_u offset=0xffff0000))

;; wasm[0]::function[0]:
;;    0: addi    sp, sp, -0x10
;;    4: sd      ra, 8(sp)
;;    8: sd      s0, 0(sp)
;;    c: mv      s0, sp
;;   10: ld      a5, 0x50(a0)
;;   14: slli    a4, a2, 0x20
;;   18: srli    a0, a4, 0x20
;;   1c: add     a5, a5, a0
;;   20: lui     a4, 0xffff
;;   24: slli    a0, a4, 4
;;   28: add     a5, a5, a0
;;   2c: sb      a3, 0(a5)
;;   30: ld      ra, 8(sp)
;;   34: ld      s0, 0(sp)
;;   38: addi    sp, sp, 0x10
;;   3c: ret
;;
;; wasm[0]::function[1]:
;;   40: addi    sp, sp, -0x10
;;   44: sd      ra, 8(sp)
;;   48: sd      s0, 0(sp)
;;   4c: mv      s0, sp
;;   50: ld      a5, 0x50(a0)
;;   54: slli    a4, a2, 0x20
;;   58: srli    a0, a4, 0x20
;;   5c: add     a5, a5, a0
;;   60: lui     a4, 0xffff
;;   64: slli    a0, a4, 4
;;   68: add     a5, a5, a0
;;   6c: lbu     a0, 0(a5)
;;   70: ld      ra, 8(sp)
;;   74: ld      s0, 0(sp)
;;   78: addi    sp, sp, 0x10
;;   7c: ret
;;! target = "riscv64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation=false -W memory64 -O static-memory-maximum-size=0 -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

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
;;    0: addi    sp, sp, -0x10
;;    4: sd      ra, 8(sp)
;;    8: sd      s0, 0(sp)
;;    c: mv      s0, sp
;;   10: ld      a4, 0x58(a0)
;;   14: addi    a4, a4, -4
;;   18: bltu    a4, a2, 0x20
;;   1c: ld      a4, 0x50(a0)
;;   20: add     a4, a4, a2
;;   24: sw      a3, 0(a4)
;;   28: ld      ra, 8(sp)
;;   2c: ld      s0, 0(sp)
;;   30: addi    sp, sp, 0x10
;;   34: ret
;;   38: .byte   0x00, 0x00, 0x00, 0x00
;;
;; wasm[0]::function[1]:
;;   3c: addi    sp, sp, -0x10
;;   40: sd      ra, 8(sp)
;;   44: sd      s0, 0(sp)
;;   48: mv      s0, sp
;;   4c: ld      a3, 0x58(a0)
;;   50: addi    a3, a3, -4
;;   54: bltu    a3, a2, 0x20
;;   58: ld      a3, 0x50(a0)
;;   5c: add     a3, a3, a2
;;   60: lw      a0, 0(a3)
;;   64: ld      ra, 8(sp)
;;   68: ld      s0, 0(sp)
;;   6c: addi    sp, sp, 0x10
;;   70: ret
;;   74: .byte   0x00, 0x00, 0x00, 0x00
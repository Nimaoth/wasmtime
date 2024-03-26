;;! target = "riscv64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -O static-memory-maximum-size=0 -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

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
;;    0: addi    sp, sp, -0x10
;;    4: sd      ra, 8(sp)
;;    8: sd      s0, 0(sp)
;;    c: mv      s0, sp
;;   10: ld      a1, 0x58(a0)
;;   14: ld      a0, 0x50(a0)
;;   18: slli    a5, a2, 0x20
;;   1c: srli    a2, a5, 0x20
;;   20: lui     a5, 1
;;   24: addi    a4, a5, 4
;;   28: sub     a1, a1, a4
;;   2c: sltu    a1, a1, a2
;;   30: add     a0, a0, a2
;;   34: lui     a2, 1
;;   38: add     a0, a0, a2
;;   3c: neg     a4, a1
;;   40: not     a1, a4
;;   44: and     a2, a0, a1
;;   48: sw      a3, 0(a2)
;;   4c: ld      ra, 8(sp)
;;   50: ld      s0, 0(sp)
;;   54: addi    sp, sp, 0x10
;;   58: ret
;;
;; wasm[0]::function[1]:
;;   5c: addi    sp, sp, -0x10
;;   60: sd      ra, 8(sp)
;;   64: sd      s0, 0(sp)
;;   68: mv      s0, sp
;;   6c: ld      a1, 0x58(a0)
;;   70: ld      a0, 0x50(a0)
;;   74: slli    a5, a2, 0x20
;;   78: srli    a2, a5, 0x20
;;   7c: lui     a5, 1
;;   80: addi    a3, a5, 4
;;   84: sub     a1, a1, a3
;;   88: sltu    a1, a1, a2
;;   8c: add     a0, a0, a2
;;   90: lui     a2, 1
;;   94: add     a0, a0, a2
;;   98: neg     a4, a1
;;   9c: not     a1, a4
;;   a0: and     a2, a0, a1
;;   a4: lw      a0, 0(a2)
;;   a8: ld      ra, 8(sp)
;;   ac: ld      s0, 0(sp)
;;   b0: addi    sp, sp, 0x10
;;   b4: ret
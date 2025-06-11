#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#define DEBUG 1
#define PROGRAM_LEN_LIMIT 400
// CPU STATE
int PC; // program counter

#define X0   0  /* Hard-wired zero */
int REGISTERS[32];
#define MEM_SIZE  (64 * 1024)         /* 64 KiB example */
uint8_t dMEM[MEM_SIZE];               /* byte-addressable memory */

// CSR address space size
#define CSR_SPACE        4096
static uint32_t CSR[CSR_SPACE];    /* Control & Status Registers */


bool skip_pc_increment = false;

// GLOBAL INSTRUCTION DECODE DATA
int opcode;
int imm;
int rs1;
int rs2;
int funct3;
int funct7;
int rd;
int instruction;

/////////////// CONVENIENCE FUNCTIONS ///////////////
// used for bitshifting
int bitarrayBuffer[32] = {0};

static const uint32_t POW2[32] = {
    0x00000001u, 0x00000002u, 0x00000004u, 0x00000008u,
    0x00000010u, 0x00000020u, 0x00000040u, 0x00000080u,
    0x00000100u, 0x00000200u, 0x00000400u, 0x00000800u,
    0x00001000u, 0x00002000u, 0x00004000u, 0x00008000u,
    0x00010000u, 0x00020000u, 0x00040000u, 0x00080000u,
    0x00100000u, 0x00200000u, 0x00400000u, 0x00800000u,
    0x01000000u, 0x02000000u, 0x04000000u, 0x08000000u,
    0x10000000u, 0x20000000u, 0x40000000u, 0x80000000u
};


static inline void resetBitarrayBuffer() {
    memset(bitarrayBuffer, 0, 32 * sizeof(int));
}

static inline void numToBits(int val) {
    resetBitarrayBuffer();
    uint32_t u = (uint32_t)val;          /* view bits as unsigned */

    for (int i = 31; i >= 0; --i) {
        if (u >= POW2[i]) {              /* test bit i */
            bitarrayBuffer[i] = 1;
            u -= POW2[i];
        }
    }
}

static inline int bitsToNum(void) {
    uint32_t u = 0;
    for (int i = 0; i < 32; ++i) {
        if (bitarrayBuffer[i]) {
            u += POW2[i];
        }
    }
    return (int32_t)u;
}

static inline int leftShift(int x, int n) {
    // in bitarrayBuffer, MSB is at index 31 (it's reversed)
    if (n <= 0)  return x;
    if (n >= 32) return 0;

    numToBits(x);
    for (int i = 31; i >= n; i--) bitarrayBuffer[i] = bitarrayBuffer[i-n];
    for (int i = 0; i < n; i++) bitarrayBuffer[i] = 0;
    return bitsToNum();
}

static inline int rightShift(int x, int n) {  
    // this is a LOGICAL right shift
    if (n <= 0) return x; // no shift
    if (n >= 32) return 0; // shift all bits out

    numToBits(x); 

    for (int i = 0; i < 32; i++) {
        if (i + n < 32) {
            bitarrayBuffer[i] = bitarrayBuffer[i+n];
        } else {  // shift in zeroes at the top
            bitarrayBuffer[i] = 0;
        }
    }

    return bitsToNum();
}

static inline int rightShiftArith(int x, int n) {  
    /* arithmetic right shift (sign-fill) */
    if (n <= 0) return x; // no shift
    if (n >= 32) return (x < 0) ? -1 : 0; // shift all bits out, fill with MSB
    
    numToBits(x); 
    int MSB = bitarrayBuffer[31];
    for (int i = 0; i < 32; i++) {
        if (i + n < 32) {
            bitarrayBuffer[i] = bitarrayBuffer[i + n];
        } else {  // shift in zeroes at the top
            bitarrayBuffer[i] = MSB;
        }
    }
    return bitsToNum();
}

static inline int getnbits(int msb, int lsb, int bits) {
    numToBits(bits); // fills bitarrayBuffer[0..31]

    int width = msb - lsb + 1;
    int resultArray[32] = {0};
    memcpy(resultArray, bitarrayBuffer + lsb, width * sizeof(int));
    memcpy(bitarrayBuffer, resultArray, 32 * sizeof(int));

    return bitsToNum();
}

static inline int bitwiseNot(int x) {
    numToBits(x);

    for (int i = 0; i < 32; ++i) {
        bitarrayBuffer[i] = 1 - bitarrayBuffer[i];
    }

    return bitsToNum();
}

static inline int bitwiseAnd(int x1, int x2) {
    // bitwise AND: result = x2 & x1
    // Start with x2’s bits
    numToBits(x2);
    int resultBits[32];
    memcpy(resultBits, bitarrayBuffer, sizeof(resultBits));

    // Overlay: clear any bit where x1 has 0
    numToBits(x1);
    for (int i = 0; i < 32; ++i) {
        if (bitarrayBuffer[i] == 0)
            resultBits[i] = 0;
    }

    memcpy(bitarrayBuffer, resultBits, sizeof(resultBits));
    return bitsToNum();
}

static inline int bitwiseOr(int x1, int x2) {
    // bitwise OR: result = x2 | x1
    // Start with x2’s bits in the buffer
    numToBits(x2);
    int resultBits[32];
    memcpy(resultBits, bitarrayBuffer, sizeof(resultBits));

    // Overlay x1’s bits
    numToBits(x1);
    for (int i = 0; i < 32; ++i) {
        if (bitarrayBuffer[i]) resultBits[i] = 1;
    }

    memcpy(bitarrayBuffer, resultBits, sizeof(resultBits));
    return bitsToNum();
}

static inline int bitwiseXor(int x1, int x2) {
    // bitwise XOR: result = x2 ^ x1
    // Start with x2’s bits
    numToBits(x2);
    int resultBits[32];
    memcpy(resultBits, bitarrayBuffer, sizeof(resultBits));

    // Overlay: flip any bit where x1 has 1
    numToBits(x1);
    for (int i = 0; i < 32; ++i) {
        if (bitarrayBuffer[i] == 1)
            resultBits[i] = 1 - resultBits[i];
    }

    memcpy(bitarrayBuffer, resultBits, sizeof(resultBits));
    return bitsToNum();
}


/////////////// EMULATOR FUNCTIONS ///////////////
#pragma region R_Type_Emulation_Functions

// 2.4.2. Integer Register-Register Operations
static inline void add_op(void)  { REGISTERS[rd] = REGISTERS[rs1] + REGISTERS[rs2]; }
static inline void sub_op(void)  { REGISTERS[rd] = REGISTERS[rs1] - REGISTERS[rs2]; }
static inline void and_op(void)  { REGISTERS[rd] = bitwiseAnd (REGISTERS[rs1], REGISTERS[rs2]); }
static inline void or_op (void)  { REGISTERS[rd] = bitwiseOr  (REGISTERS[rs1], REGISTERS[rs2]); }
static inline void xor_op(void)  { REGISTERS[rd] = bitwiseXor (REGISTERS[rs1], REGISTERS[rs2]); }

static inline int shamt_from_reg(int r) { return getnbits(4, 0, REGISTERS[r]); }

static inline void sll_op(void)  { REGISTERS[rd] = leftShift       (REGISTERS[rs1], shamt_from_reg(rs2)); }
static inline void srl_op(void)  { REGISTERS[rd] = rightShift      (REGISTERS[rs1], shamt_from_reg(rs2)); }
static inline void sra_op(void)  { REGISTERS[rd] = rightShiftArith (REGISTERS[rs1], shamt_from_reg(rs2)); }


static inline void slt_op(void)  {
    REGISTERS[rd] = (REGISTERS[rs1] < REGISTERS[rs2]) ? 1 : 0;
}

static inline void sltu_op(void) {
    unsigned int u1 = REGISTERS[rs1];
    unsigned int u2 = REGISTERS[rs2];
    REGISTERS[rd] = (u1 < u2) ? 1 : 0;
}
#pragma endregion

#pragma region I_Type_Emulation_Functions

// --- helper: sign-extend 12-bit immediate ---------------------------------
static inline int sign_extend_12(int imm12) {        // 0x800 = bit11 set
    return (imm12 >= 0x800) ? (imm12 - 0x1000) : imm12;
}

// 2.4.1. Integer Register-Immediate Instructions
static inline void addi_op(void) {
    REGISTERS[rd] = REGISTERS[rs1] + sign_extend_12(imm);
}

static inline void slti_op(void) {       // signed compare
    int simm = sign_extend_12(imm);
    REGISTERS[rd] = (REGISTERS[rs1] < simm) ? 1 : 0;
}

static inline void sltiu_op(void) {      // unsigned compare
    unsigned int u1 = REGISTERS[rs1];
    unsigned int uimm = (unsigned)sign_extend_12(imm);
    REGISTERS[rd] = (u1 < uimm) ? 1 : 0;
}

static inline void ori_op(void)  { REGISTERS[rd] = bitwiseOr (REGISTERS[rs1], sign_extend_12(imm)); }
static inline void andi_op(void) { REGISTERS[rd] = bitwiseAnd(REGISTERS[rs1], sign_extend_12(imm)); }
static inline void xori_op(void) { REGISTERS[rd] = bitwiseXor(REGISTERS[rs1], sign_extend_12(imm)); }

// --- Shift-immediate ops : shamt = imm[4:0] (zero-extended) ---------------
static inline int shamt5(void) { return getnbits(4, 0, imm); }

static inline void slli_op(void) { REGISTERS[rd] = leftShift        (REGISTERS[rs1], shamt5()); }
static inline void srli_op(void) { REGISTERS[rd] = rightShift       (REGISTERS[rs1], shamt5()); }
static inline void srai_op(void) { REGISTERS[rd] = rightShiftArith  (REGISTERS[rs1], shamt5()); }

// 2.5.1. Unconditional Jumps
static inline void jalr_op(void) {
    int simm   = sign_extend_12(imm);
    int target = REGISTERS[rs1] + simm;

    /* Clear bits 1:0 to enforce 4-byte alignment for RV32 */
    numToBits(target);  
    bitarrayBuffer[0] = 0;  
    bitarrayBuffer[1] = 0;  
    target = bitsToNum();

    REGISTERS[rd] = PC + 4;
    PC = target;                             
    skip_pc_increment = true;
}

// 2.6. Load and Store Instructions
/* Trusted program: no mem-fault check */
static inline int load_half(int addr) {
    /* Assemble a 16-bit half-word from two bytes at addr and addr+1. */
    int b0 = dMEM[addr];           /* low  byte */
    int b1 = dMEM[addr + 1];       /* high byte */
    return b0 + leftShift(b1, 8);  /* b1 << 8 without << */
}

static inline int load_word(int addr) {
    /* Assemble a 32-bit word from four bytes. */
    int b0 = dMEM[addr];
    int b1 = dMEM[addr + 1];
    int b2 = dMEM[addr + 2];
    int b3 = dMEM[addr + 3];
    return  b0
          + leftShift(b1,  8)
          + leftShift(b2, 16)
          + leftShift(b3, 24);
}

static inline void lb_op(void) {
    /* Load Byte (signed) */
    int addr = REGISTERS[rs1] + sign_extend_12(imm);
    int byte = dMEM[addr];
    REGISTERS[rd] = (byte >= 0x80) ? (byte - 0x100) : byte;   /* sign-extend 8→32 */
}

static inline void lh_op(void) {
    /* Load Half (signed) */
    int addr = REGISTERS[rs1] + sign_extend_12(imm);
    int half = load_half(addr);
    REGISTERS[rd] = (half >= 0x8000) ? (half - 0x10000) : half; /* sign-extend 16→32 */
}

static inline void lw_op(void) {
    /* Load Word */
    int addr = REGISTERS[rs1] + sign_extend_12(imm);
    REGISTERS[rd] = load_word(addr);
}

static inline void lbu_op(void) {
    int addr = REGISTERS[rs1] + sign_extend_12(imm);
    REGISTERS[rd] = dMEM[addr];
}

static inline void lhu_op(void) {
    int addr = REGISTERS[rs1] + sign_extend_12(imm);
    REGISTERS[rd] = load_half(addr);
}

// 2.7. Memory Ordering Instructions
static inline void fence_op(int fm, int pred, int succ)
{
    /* currently as nop */
}

static inline void fence_tso_op(void)   { /* nop */ }
static inline void pause_op(void)       { /* nop */ }

// 2.8. Environment Call and Breakpoints
static inline void ecall_op(void) {
    if (DEBUG) printf("ECALL at PC=%u → exiting\n", (unsigned)PC);
    skip_pc_increment = true;
    PC = PROGRAM_LEN_LIMIT;          /* forces main loop to exit */
}

static inline void ebreak_op(void) {
    if (DEBUG) printf("EBREAK at PC=%u → exiting\n", (unsigned)PC);
    skip_pc_increment = true;
    PC = PROGRAM_LEN_LIMIT;
}

// Chapter 5. "Zifencei" Extension for Instruction-Fetch Fence,
static inline void fence_i_op(void) {
    /* Zifencei: instruction-fetch barrier (no-op in single-core ISS) */
}

// Chapter 6. "Zicsr", Extension for Control and Status Register (CSR) Instructions,

static inline void csrrw_op(void) {
    // CSRRW: Atomic Read/Write CSR
    //   if rd != x0: rd = CSR; CSR = rs1
    //   if rd == x0: skip the read (no side-effects), still write CSR = rs1
    int csr_index = imm; // getnbits(31, 20, instruction);
    uint32_t tmp = (uint32_t)REGISTERS[rs1];
    if (rd != X0) {
        uint32_t old = CSR[csr_index];
        REGISTERS[rd] = old;        // zero-extended read
    }
    CSR[csr_index] = tmp;        // write always happens
}

static inline void csrrs_op(void) {
    // CSRRS: Atomic Read and Set Bits in CSR
    //   always read→rd, then if rs1 != x0, CSR |= rs1
    int csr_index = imm; // getnbits(31, 20, instruction);

    uint32_t old = CSR[csr_index];
    REGISTERS[rd] = old;            // always read
    if (rs1 != X0) {
        // compute old | rs1
        uint32_t mask = (uint32_t)REGISTERS[rs1];
        CSR[csr_index] = bitwiseOr(old, mask);
    }
}

static inline void csrrc_op(void) {
    // CSRRC: Atomic Read and Clear Bits in CSR
    //   always read→rd, then if rs1 != x0, CSR &= ~rs1
    int csr_index = imm; // getnbits(31, 20, instruction);

    uint32_t old = CSR[csr_index];
    REGISTERS[rd] = old;            // always read

    if (rs1 != X0) {
        uint32_t mask = (uint32_t)REGISTERS[rs1];
        CSR[csr_index] = bitwiseAnd(old, bitwiseNot(mask));
    }
}

static inline void csrrwi_op(void) {
    // CSRRWI: Atomic Read/Write Immediate
    //   if rd != x0: rd = CSR; CSR = uimm
    //   if rd == x0: skip read, still write
    int csr_index = imm; // getnbits(31, 20, instruction);
    uint32_t uimm = (uint32_t)rs1;      // getnbits(19,15,instruction);
    uint32_t tmp = (uint32_t)uimm;
    if (rd != X0) {
        uint32_t old = CSR[csr_index];
        REGISTERS[rd] = old;
    }
    CSR[csr_index] = tmp;
}

static inline void csrrsi_op(void) {
    // CSRRSI: Atomic Read and Set Immediate
    //   always read→rd, then if uimm != 0, CSR |= uimm
    int csr_index = imm; // getnbits(31, 20, instruction);
    uint32_t uimm = (uint32_t)rs1;      // getnbits(19,15,instruction);
    uint32_t old = CSR[csr_index];
    REGISTERS[rd] = old;
    if (uimm != 0) {
        // OR with immediate mask
        CSR[csr_index] = bitwiseOr(old, uimm);
    }
}

static inline void csrrci_op(void) {
    // CSRRCI: Atomic Read and Clear Immediate
    //   always read→rd, then if uimm != 0, CSR &= ~uimm
    int csr_index = imm; // getnbits(31, 20, instruction);
    uint32_t uimm = (uint32_t)rs1;      // getnbits(19,15,instruction);
    uint32_t old = CSR[csr_index];
    REGISTERS[rd] = old;
    if (uimm != 0) {
        CSR[csr_index] = bitwiseAnd(old, bitwiseNot(uimm));
    }
}

// Chapter 12. M Extension for Integer Multiplication and Division,
static inline void mul_op(void) {
    int64_t r = (int64_t)(int32_t)REGISTERS[rs1] * (int64_t)(int32_t)REGISTERS[rs2];
    REGISTERS[rd] = (int32_t)r;
}

// TODO: fix i64 >>
static inline void mulh_op(void) {
    int64_t r = (int64_t)(int32_t)REGISTERS[rs1] * (int64_t)(int32_t)REGISTERS[rs2];
    REGISTERS[rd] = (int32_t)(r >> 32);
}

static inline void mulhsu_op(void) {
    int64_t r = (int64_t)(int32_t)REGISTERS[rs1] *
                                  (int64_t)(uint32_t)REGISTERS[rs2];
    REGISTERS[rd] = (int32_t)(r >> 32);
}

static inline void mulhu_op(void) {
    uint64_t r = (uint64_t)(uint32_t)REGISTERS[rs1] * (uint64_t)(uint32_t)REGISTERS[rs2];
    REGISTERS[rd] = (int32_t)(r >> 32);
}

// TODO: fix / if not supported
static inline void div_op(void) {
    int32_t a = (int32_t)REGISTERS[rs1];
    int32_t b = (int32_t)REGISTERS[rs2];
    REGISTERS[rd] = (b == 0 ? -1 : a / b);
}

static inline void divu_op(void) {
    uint32_t a = (uint32_t)REGISTERS[rs1];
    uint32_t b = (uint32_t)REGISTERS[rs2];
    REGISTERS[rd] = (b == 0 ? UINT32_MAX : a / b);
}

// TODO: fix % if not supported
static inline void rem_op(void) {
    int32_t a = (int32_t)REGISTERS[rs1];
    int32_t b = (int32_t)REGISTERS[rs2];
    REGISTERS[rd] = (b == 0 ? a : a % b);
}

static inline void remu_op(void) {
    uint32_t a = (uint32_t)REGISTERS[rs1];
    uint32_t b = (uint32_t)REGISTERS[rs2];
    REGISTERS[rd] = (b == 0 ? a : a % b);
}

#pragma endregion

#pragma region U_Type_Emulation_Functions

static inline int sign_extend_20(int imm20){
    return (imm20 >= 0x80000) ? (imm20 - 0x100000) : imm20;
}

//2.4.1. Integer Register-Immediate Instructions
static inline void lui_op(void) {
    int imm20 = sign_extend_20(imm);          // imm is global 20-bit field
    REGISTERS[rd] = leftShift(imm20, 12);     // imm[31:12] << 12
}

static inline void auipc_op(void) {
    int imm20 = sign_extend_20(imm);
    REGISTERS[rd] = PC + leftShift(imm20, 12);
}

#pragma endregion

#pragma region J_Type_Emulation_Functions
// 2.5.1. Unconditional Jumps

// 0x1_00000 = 2^20  (bit 20 set)
// 0x2_00000 = 2^21
static inline int sign_extend_21(int off21)
{
    return (off21 >= 0x100000) ? (off21 - 0x200000) : off21;
}

static inline void jal_op(void) {
    REGISTERS[rd] = PC + 4; 
    PC += sign_extend_21(imm);
    skip_pc_increment = true; 
}

#pragma endregion

#pragma region S_Type_Emulation_Functions

// 2.6. Load and Store Instructions
static inline void sb_op(void) {      
    /* Store Byte */
    int addr = REGISTERS[rs1] + sign_extend_12(imm);
    dMEM[addr] = getnbits(7, 0, REGISTERS[rs2]);
}

static inline void store_half(int addr, int value16) {
    /* Store a 16-bit value into two bytes. */
    dMEM[addr]     = getnbits(7,  0, value16);
    dMEM[addr + 1] = getnbits(15, 8, value16);
}

static inline void sh_op(void) {      
    /* Store Half-word */
    int addr = REGISTERS[rs1] + sign_extend_12(imm);
    store_half(addr, getnbits(15, 0, REGISTERS[rs2]));
}

static inline void store_word(int addr, int value32) {
    /* Store a 32-bit value into four bytes. */
    dMEM[addr    ] = getnbits( 7,  0, value32);
    dMEM[addr + 1] = getnbits(15,  8, value32);
    dMEM[addr + 2] = getnbits(23, 16, value32);
    dMEM[addr + 3] = getnbits(31, 24, value32);
}

static inline void sw_op(void) {
    /* Store Word */
    int addr = REGISTERS[rs1] + sign_extend_12(imm);
    store_word(addr, REGISTERS[rs2]);
}


#pragma endregion

#pragma region B_Type_Emulation_Functions
// 0x1000 =  2^12 (bit 12 set)
// 0x2000 =  2^13
static inline int sign_extend_13(int off13) {
    return (off13 >= 0x1000) ? (off13 - 0x2000) : off13;
}

// 2.5.2. Conditional Branches

static inline void beq_op(void) {
    if (REGISTERS[rs1] == REGISTERS[rs2]) {
        PC += sign_extend_13(imm);
        skip_pc_increment = true;
    }
}

static inline void bne_op(void) {
    if (REGISTERS[rs1] != REGISTERS[rs2]) {
        PC += sign_extend_13(imm);
        skip_pc_increment = true;
    }
}

static inline void blt_op(void) {
    if (REGISTERS[rs1] < REGISTERS[rs2]) {
        PC += sign_extend_13(imm);
        skip_pc_increment = true;
    }
}

static inline void bge_op(void) {
    if (REGISTERS[rs1] >= REGISTERS[rs2]) {
        PC += sign_extend_13(imm);
        skip_pc_increment = true;
    }
}

static inline void bltu_op(void) {
    unsigned int u1 = REGISTERS[rs1];
    unsigned int u2 = REGISTERS[rs2];
    if (u1 < u2) {
        PC += sign_extend_13(imm);
        skip_pc_increment = true;
    }
}

static inline void bgeu_op(void) {
    unsigned int u1 = REGISTERS[rs1];
    unsigned int u2 = REGISTERS[rs2];
    if (u1 >= u2) {
        PC += sign_extend_13(imm);
        skip_pc_increment = true;
    }
}

#pragma endregion

/////////////// DECODE LOGIC ///////////////

#pragma region R_Type_Instruction_Encodings
// === R-Type  ===
// === R-type opcode ===
#define OPCODE_R_TYPE   0b0110011

// ── funct7 groups ─────────────────────────────────────────────────────────
// Base integer Register-Register ops
#define FUNCT7_BASE     0b0000000
// SUB and SRA (same funct3, different semantics)
#define FUNCT7_SUB_SRA  0b0100000
// M-extension (RV32M)
#define FUNCT7_M_EXT    0b0000001

// ── funct3 for FUNCT7_BASE ────────────────────────────────────────────────
#define F3_ADD          0b000    // ADD
#define F3_SLL          0b001    // SLL
#define F3_SLT          0b010    // SLT
#define F3_SLTU         0b011    // SLTU
#define F3_XOR          0b100    // XOR
#define F3_SRL          0b101    // SRL
#define F3_OR           0b110    // OR
#define F3_AND          0b111    // AND

// ── funct3 for FUNCT7_SUB_SRA ─────────────────────────────────────────────
#define F3_SUB          0b000    // SUB (when funct7==0100000)
#define F3_SRA          0b101    // SRA (when funct7==0100000)

// ── funct3 for FUNCT7_M_EXT (RV32M) ───────────────────────────────────────
#define F3_MUL          0b000
#define F3_MULH         0b001
#define F3_MULHSU       0b010
#define F3_MULHU        0b011
#define F3_DIV          0b100
#define F3_DIVU         0b101
#define F3_REM          0b110
#define F3_REMU         0b111

#pragma endregion

void decode_R_type(void) {
    // func7 (31 to 25) rs2(24 to 20) rs1(19 to 15) funct3(14 to 12) rd (11 to 7) opcode (6 to 0)
    funct7 = getnbits(31, 25, instruction);
    rs2    = getnbits(24, 20, instruction);
    rs1    = getnbits(19, 15, instruction);
    funct3 = getnbits(14, 12, instruction);
    rd     = getnbits(11, 7, instruction);

    // 2.4.2. Integer Register-Register Operations
    if (funct7 == FUNCT7_BASE) {
        // Base R-type (funct7 = 0000000)
        if      (funct3 == F3_ADD ) add_op();
        else if (funct3 == F3_SLL ) sll_op();
        else if (funct3 == F3_SLT ) slt_op();
        else if (funct3 == F3_SLTU) sltu_op();
        else if (funct3 == F3_XOR ) xor_op();
        else if (funct3 == F3_SRL ) srl_op();
        else if (funct3 == F3_OR  ) or_op();
        else if (funct3 == F3_AND ) and_op();
    }
    else if (funct7 == FUNCT7_SUB_SRA) {
        // SUB / SRA (funct7 = 0100000)
        if      (funct3 == F3_SUB) sub_op();
        else if (funct3 == F3_SRA) sra_op();
    }
    else if (funct7 == FUNCT7_M_EXT) {
        // Chapter 12. M Extension for Integer Multiplication and Division,
        // M-extension (funct7 = 0000001)
        if      (funct3 == F3_MUL   ) mul_op();
        else if (funct3 == F3_MULH  ) mulh_op();
        else if (funct3 == F3_MULHSU) mulhsu_op();
        else if (funct3 == F3_MULHU ) mulhu_op();
        else if (funct3 == F3_DIV   ) div_op();
        else if (funct3 == F3_DIVU  ) divu_op();
        else if (funct3 == F3_REM   ) rem_op();
        else if (funct3 == F3_REMU  ) remu_op();
    }
        // Illegal R-type
    else {
        if (DEBUG) printf("Illegal R-type funct7=0b%07b at PC=%u\n", funct7, (unsigned)PC);
        skip_pc_increment = true;
        PC = PROGRAM_LEN_LIMIT;
    }
}

#pragma region I_Type_Instruction_Encodings
// === I-Type ===

// 2.4.1. Integer Register-Immediate Instructions
#define OPCODE_I_ARITH    0b0010011
#define FUNCT3_ADDI       0b000
#define FUNCT3_SLTI       0b010
#define FUNCT3_SLTIU      0b011
#define FUNCT3_XORI       0b100
#define FUNCT3_ORI        0b110
#define FUNCT3_ANDI       0b111
#define FUNCT3_SLLI       0b001
#define FUNCT3_SRLI_SRAI  0b101

#define FUNCT7_SLLI       0b0000000
#define FUNCT7_SRLI       0b0000000
#define FUNCT7_SRAI       0b0100000

// 2.5.1. Unconditional Jumps
#define OPCODE_I_JALR     0b1100111
#define FUNCT3_JALR   0b000     

// 2.6. Load and Store Instructions
#define OPCODE_I_LOAD     0b0000011
#define FUNCT3_LB         0b000
#define FUNCT3_LH         0b001
#define FUNCT3_LW         0b010
#define FUNCT3_LBU        0b100
#define FUNCT3_LHU        0b101

// 2.7. Memory Ordering Instructions
#define OPCODE_MISC_MEM   0b0001111     /* FENCE, FENCE.TSO, PAUSE   */
#define FUNCT3_FENCE      0b000

/* fm pred succ  → imm[11:0] */
#define IMM12_FENCE       0b000000000000  /* fm=0000 pred=0000 succ=0000 */
#define IMM12_FENCE_TSO   0b100000110011  /* fm=1000 pred=0011 succ=0011 */
#define IMM12_PAUSE       0b000000010000  /* fm=0000 pred=0001 succ=0000 */

// 2.8. Environment Call and Breakpoints
#define OPCODE_I_SYSTEM   0b1110011

#define FUNCT3_SYSTEM    0b000           /* ECALL / EBREAK */
#define IMM12_ECALL      0x000
#define IMM12_EBREAK     0x001

// Chapter 5. "Zifencei" Extension for Instruction-Fetch Fence,
#define FUNCT3_FENCE_I   0b001  /* FENCE.I */

// Chapter 6. "Zicsr", Extension for Control and Status Register (CSR) Instructions,
 /* opcode same as SYSTEM opcode */

// funct3 values
#define FUNCT3_CSRRW     0b001       /* Atomic read/write   */
#define FUNCT3_CSRRS     0b010       /* Atomic read/set     */
#define FUNCT3_CSRRC     0b011       /* Atomic read/clear   */
#define FUNCT3_CSRRWI    0b101
#define FUNCT3_CSRRSI    0b110
#define FUNCT3_CSRRCI    0b111


#pragma endregion

void decode_I_type(void){
    imm     = getnbits(31, 20, instruction);
    rs1     = getnbits(19, 15, instruction);
    funct3  = getnbits(14, 12, instruction);
    rd      = getnbits(11, 7, instruction);

    if (opcode == OPCODE_I_ARITH) {
        // 2.4.1. Integer Register-Immediate Instructions
        if      (funct3 == FUNCT3_ADDI)  addi_op();
        else if (funct3 == FUNCT3_XORI)  xori_op();
        else if (funct3 == FUNCT3_ORI)   ori_op();
        else if (funct3 == FUNCT3_ANDI)  andi_op();
        else if (funct3 == FUNCT3_SLTI)  slti_op();
        else if (funct3 == FUNCT3_SLTIU) sltiu_op();
        else if (funct3 == FUNCT3_SLLI) {
            int f7 = getnbits(31, 25, instruction);
            if (f7 == FUNCT7_SLLI) slli_op();
        } else if (funct3 == FUNCT3_SRLI_SRAI) {
            int f7 = getnbits(31, 25, instruction);
            if      (f7 == FUNCT7_SRLI) srli_op();
            else if (f7 == FUNCT7_SRAI) srai_op();
        }
    }
    else if (opcode == OPCODE_I_JALR && funct3 == FUNCT3_JALR) {
        // 2.5.1. Unconditional Jumps
        jalr_op();
    }
    else if (opcode == OPCODE_I_LOAD) {
        // 2.6. Load and Store Instructions
        if      (funct3 == FUNCT3_LB)  lb_op();
        else if (funct3 == FUNCT3_LH)  lh_op();
        else if (funct3 == FUNCT3_LW)  lw_op();
        else if (funct3 == FUNCT3_LBU) lbu_op();
        else if (funct3 == FUNCT3_LHU) lhu_op();
    }
    else if (opcode == OPCODE_MISC_MEM) {
        if (funct3 == FUNCT3_FENCE) {
            // 2.7. Memory Ordering Instructions
            int succ = getnbits(3,  0, imm);            /* imm[3:0]   */
            int pred = getnbits(7,  4, imm);            /* imm[7:4]   */
            int fm   = getnbits(11, 8, imm);            /* imm[11:8]  */
            if      (imm == IMM12_FENCE)       fence_op(fm, pred, succ);
            else if (imm == IMM12_FENCE_TSO)   fence_tso_op();
            else if (imm == IMM12_PAUSE)       pause_op();
            else                               fence_op(fm, pred, succ); /* other hints */

        } else if (funct3 == FUNCT3_FENCE_I) {
            // Chapter 5. "Zifencei" Extension for Instruction-Fetch Fence,
            fence_i_op();
        }

    }
    else if (opcode == OPCODE_I_SYSTEM) {
        if (funct3 == FUNCT3_SYSTEM) {
            // 2.8. Environment Call and Breakpoints
            int sys_imm = getnbits(31, 20, instruction);
            if      (sys_imm == IMM12_ECALL)  ecall_op();
            else if (sys_imm == IMM12_EBREAK) ebreak_op();
        } else {
            // Chapter 6. "Zicsr", Extension for Control and Status Register (CSR) Instructions,
            if      (funct3 == FUNCT3_CSRRW)   csrrw_op();
            else if (funct3 == FUNCT3_CSRRS)   csrrs_op();
            else if (funct3 == FUNCT3_CSRRC)   csrrc_op();
            else if (funct3 == FUNCT3_CSRRWI)  csrrwi_op();
            else if (funct3 == FUNCT3_CSRRSI)  csrrsi_op();
            else if (funct3 == FUNCT3_CSRRCI)  csrrci_op();
        }

    }
    else {
        printf("ERROR: Unknown I-type instruction with opcode %d and funct3 %d\n", opcode, funct3);
        skip_pc_increment = true;
        PC = PROGRAM_LEN_LIMIT;
        return;
    }
}

#pragma region U_Type_Instruction_Encodings
// === U-Type (LUI / AUIPC) ===
#define OPCODE_LUI     0b0110111   // 55
#define OPCODE_AUIPC   0b0010111   // 23

#pragma endregion

void decode_U_type(void) {
    // imm(31 to 12) rd(11 to 7) opcode (6 to 0)
    imm = getnbits(31, 12, instruction);
    rd = getnbits(11, 7, instruction);
    // 2.4.1. Integer Register-Immediate Instructions
    if (opcode == OPCODE_LUI) {
        lui_op();
    } else if (opcode == OPCODE_AUIPC) {
        auipc_op();
    }
}

#pragma region J_Type_Instruction_Encodings
// === J-Type ===
#define OPCODE_JAL  0b1101111   /* 111 decimal */
#pragma endregion

void decode_J_type(void) {
    /* imm[20|10:1|11|19:12]  + bit 0 = 0 (word-aligned)
    Compose 21-bit offset, then jal_op handles sign-ext */
    imm = leftShift(getnbits(31,31, instruction), 20)   // bit 20 (sign)
        + leftShift(getnbits(19,12, instruction), 12)   // bits 19-12
        + leftShift(getnbits(20,20, instruction), 11)   // bit 11
        + leftShift(getnbits(30,21, instruction), 1);   // bits 10-1, LSB zero

    rd = getnbits(11,7,instruction);
    // 2.5.1. Unconditional Jumps
    jal_op();
}

#pragma region B_Type_Instruction_Encodings
#define OPCODE_BRANCH  0b1100011  
#define FUNCT3_BEQ   0b000
#define FUNCT3_BNE   0b001
#define FUNCT3_BLT   0b100
#define FUNCT3_BGE   0b101
#define FUNCT3_BLTU  0b110
#define FUNCT3_BGEU  0b111

#pragma endregion

void decode_B_type(void) {
    /* 13-bit offset layout: imm[12|10:5|4:1|11] and implicit bit-0 = 0. */
    imm = leftShift(getnbits(31,31,instruction), 12)   /* imm[12]   */
        + leftShift(getnbits(30,25,instruction),  5)   /* imm[10:5] */
        + leftShift(getnbits(11, 8,instruction),  1)   /* imm[4:1]  */
        + leftShift(getnbits(7, 7, instruction), 11);  /* imm[11]   */

    rs2    = getnbits(24,20,instruction);
    rs1    = getnbits(19,15,instruction);
    funct3 = getnbits(14,12,instruction);

    // 2.5.2. Conditional Branches
    if      (funct3 == FUNCT3_BEQ)   beq_op();
    else if (funct3 == FUNCT3_BNE)   bne_op();
    else if (funct3 == FUNCT3_BLT)   blt_op();
    else if (funct3 == FUNCT3_BGE)   bge_op();
    else if (funct3 == FUNCT3_BLTU)  bltu_op();
    else if (funct3 == FUNCT3_BGEU)  bgeu_op();
}

#pragma region S_Type_Instruction_Encodings

#define OPCODE_STORE  0b0100011   /* 35 decimal */

#define FUNCT3_SB     0b000
#define FUNCT3_SH     0b001
#define FUNCT3_SW     0b010

#pragma endregion

void decode_S_type(void) {
    /* imm = imm[11:5] | imm[4:0] */
    imm =  leftShift(getnbits(31,25,instruction), 5)   /* imm[11:5] */
        +            getnbits(11, 7,instruction);      /* imm[4:0]  */

    rs2    = getnbits(24,20,instruction);
    rs1    = getnbits(19,15,instruction);
    funct3 = getnbits(14,12,instruction);

    // 2.6. Load and Store Instructions
    if      (funct3 == FUNCT3_SB) sb_op();
    else if (funct3 == FUNCT3_SH) sh_op();
    else if (funct3 == FUNCT3_SW) sw_op();
}

void init_decode() {
    // Initialize global variables used in decode
    opcode = 0;
    imm = 0;
    rs1 = 0;
    rs2 = 0;
    funct3 = 0;
    funct7 = 0;
    rd = 0;
    skip_pc_increment = false;
}

void debug_info_step(void) {
    printf("instruction  %d\n", instruction);
    printf("rd           %d\n", rd);
    printf("rs1: %2d = %10u\n", rs1, REGISTERS[rs1]);
    printf("rs2: %2d = %10u\n", rs2, REGISTERS[rs2]);
    printf("funct3       %d\n",  funct3);
    printf("opcode       %d\n",  opcode);
    printf("imm          %d\n",  imm);
    printf("RESULT       %u\n",  REGISTERS[rd]);
    printf("PC AFTER     %u\n\n", (unsigned)PC);
}

void decode(void) {
    // Types: R, I, S, B, U, J
    opcode = getnbits(6, 0, instruction);

    if (opcode == OPCODE_R_TYPE) { 
        decode_R_type();
    } else if (opcode == OPCODE_I_ARITH || opcode == OPCODE_I_LOAD || opcode == OPCODE_I_JALR || opcode == OPCODE_I_SYSTEM) {
        decode_I_type();
    } else if (opcode == OPCODE_STORE){
        decode_S_type();
    } else if (opcode == OPCODE_BRANCH) {          /* B-type */
        decode_B_type();
    } else if (opcode == OPCODE_LUI || opcode == OPCODE_AUIPC) {   // U-type
        decode_U_type();
    } else if (opcode == OPCODE_JAL) {         /* J-type */
        decode_J_type();
    } else {
        if (DEBUG) printf("Illegal opcode 0x%X at PC=%u\n", opcode, (unsigned)PC);
        skip_pc_increment = true;
        PC = PROGRAM_LEN_LIMIT;
    }

    REGISTERS[0] = 0;  // x0 is always zero

    if (DEBUG) debug_info_step();
}

void debug_info_registers(void){
    printf("==== REGISTER DUMP ====\n");
    for (int i = 0; i < 8; ++i)
        printf("x%-2d = %u\n", i+1, REGISTERS[i+1]);
}

#define ECALL_INST 0b1110011


int main() {
    int instructions[5] = {
      8586003,
      1213075,
      6480563,
      9618195,
      115
    };

    PC = 0;
    int pc_word = 0;
    while(PC < PROGRAM_LEN_LIMIT) {
        
        instruction = instructions[pc_word];
        if (instruction == ECALL_INST) break;   /* graceful halt */

        init_decode();  // reset global variables used in decode
        decode();


        if (!skip_pc_increment) {
            PC += 4;
            ++pc_word;                        /* next sequential word */
        } else {
            pc_word = rightShift(PC, 2);      /* recompute after jump/branch */
        }
    }

    if (DEBUG) debug_info_registers();  // print final register state

}

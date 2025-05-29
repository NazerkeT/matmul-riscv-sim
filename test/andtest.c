#include <iostream>
#include <cassert>

// CPU STATE
int PC; // program counter
int REGISTERS[32];
// int iMEM[16384]; // 2^(16), read-only (ROM)
int dMEM[16384]; // 2^(16), read/write (RAM)
bool skip_pc_increment = false;
// bool quit = false;

// GLOBAL INSTRUCTION DECODE DATA
int opcode;
int imm;
int rs1;
int rs2;
int funct3;
int funct7;
int rd;
int instruction;

int bitarrayBuffer[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

int powersOfTwo[32] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256,
    512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
    262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216,
    33554432, 67108864, 134217728, 268435456, 536870912, 1073741824,
    -2147483648
};

int resetBitarrayBuffer() {
    for (int i = 0; i < 32; i++) {
        bitarrayBuffer[i] = 0;
    }
    return 0;
}

int numToBits(int val) {
    // populates result in bitarrayBuffer
    // NOTE: MSB is at index 31 (it's reversed)
    resetBitarrayBuffer();
    
    if (val < 0) {
        bitarrayBuffer[31] = 1;
        val += powersOfTwo[31];
    }

    for (int i = 30; i >= 0; i--) {
        if (val >= powersOfTwo[i]) {
            bitarrayBuffer[i] = 1;
            val -= powersOfTwo[i];
        }
    }

    return 0;
}

int bitsToNum() {
    int reconstruct = 0;
    if (bitarrayBuffer[31] == 1) {
        reconstruct = 0-powersOfTwo[31];
    }
    
    for (int i=0; i<31; i++) {
        if (bitarrayBuffer[i] == 1) {
            reconstruct += powersOfTwo[i];
        }
    }

    return reconstruct;
}

int bitwiseAnd(int x1, int x2) {
    // computes bitwise AND between x1 and x2
    // (both 32 bit integers)

    // convert both x1 and x2 to their bit represenations in the bit array buffers
    numToBits(x1);
    int x1BitArray[32];
    memcpy(x1BitArray, bitarrayBuffer, 32 * sizeof(int));

    resetBitarrayBuffer();

    numToBits(x2);
    int x2BitArray[32];
    memcpy(x2BitArray, bitarrayBuffer, 32 * sizeof(int));

    // initialize result
    resetBitarrayBuffer();
    int resultArray[32];
    memcpy(resultArray, bitarrayBuffer, 32 * sizeof(int));
    
    for (int bitIdx = 0; bitIdx < 32; bitIdx++) {
        
        // if both x1 and x2 at bitIdx are 1, set that bit in the result to 1 (bitwise and)
        if(x1BitArray[bitIdx] == 1) {
            if(x2BitArray[bitIdx] == 1) {
                resultArray[bitIdx] = 1;
            }
        }
        
    }

    resetBitarrayBuffer();
    memcpy(bitarrayBuffer, resultArray, 32 * sizeof(int));

    int result = bitsToNum();

    return result;
}

int bitwiseOr(int x1, int x2) {
    // computes bitwise OR between x1 and x2
    // (both 32 bit integers)

    // convert both x1 and x2 to their bit represenations in the bit array buffers
    numToBits(x1);
    int x1BitArray[32];
    memcpy(x1BitArray, bitarrayBuffer, 32 * sizeof(int));

    resetBitarrayBuffer();

    numToBits(x2);
    int x2BitArray[32];
    memcpy(x2BitArray, bitarrayBuffer, 32 * sizeof(int));

    // initialize result
    resetBitarrayBuffer();
    int resultArray[32];
    memcpy(resultArray, bitarrayBuffer, 32 * sizeof(int));

    for(int bitIdx = 0; bitIdx < 32; bitIdx++) {
        if(x1BitArray[bitIdx] == 1) {
            resultArray[bitIdx] = 1;
        } 
        else if(x2BitArray[bitIdx] == 1) {
            resultArray[bitIdx] = 1;
        }
    }

    resetBitarrayBuffer();
    memcpy(bitarrayBuffer, resultArray, 32 * sizeof(int));

    int result = bitsToNum();

    return result;
}

int main() {

    int num1 = 8;
    int num2 = 1;
    int res1 = 8 | 1;
    int res2 = bitwiseOr(num1, num2);

    printf("result of 8 | 1: %d\n", res1);
    printf("result of bitwiseOr(): %d\n", res2);

    return 0;
}
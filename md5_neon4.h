#include <iostream>
#include <string>
#include <cstring>
#include <arm_neon.h>
#include <vector>
using namespace std;

// 四条口令并行版本

// 定义了Byte，便于使用
typedef unsigned char Byte;
// 定义了32比特
typedef unsigned int bit32;

// MD5的一系列参数。参数是固定的，其实你不需要看懂这些
#define s11 7
#define s12 12
#define s13 17
#define s14 22
#define s21 5
#define s22 9
#define s23 14
#define s24 20
#define s31 4
#define s32 11
#define s33 16
#define s34 23
#define s41 6
#define s42 10
#define s43 15
#define s44 21



#define F_NEON(x, y, z) vorrq_u32(vandq_u32(x, y), vandq_u32(vmvnq_u32(x), z))
#define G_NEON(x, y, z) vorrq_u32(vandq_u32(x, z), vandq_u32(y, vmvnq_u32(z)))
#define H_NEON(x, y, z) veorq_u32(veorq_u32(x, y), z)
#define I_NEON(x, y, z) veorq_u32(y, vorrq_u32(x, vmvnq_u32(z)))

inline uint32x4_t ROTATELEFT_NEON(uint32x4_t num, int n) {
    return vorrq_u32(vshlq_n_u32(num, n), vshrq_n_u32(num, 32 - n));
}

#define FF_NEON(a, b, c, d, x, s, ac) { \
    a = vaddq_u32(a, vaddq_u32(F_NEON(b, c, d), vaddq_u32(x, vdupq_n_u32(ac)))); \
    a = ROTATELEFT_NEON(a, s); \
    a = vaddq_u32(a, b); \
}
#define GG_NEON(a, b, c, d, x, s, ac) { \
    a = vaddq_u32(a, vaddq_u32(G_NEON(b, c, d), vaddq_u32(x, vdupq_n_u32(ac)))); \
    a = ROTATELEFT_NEON(a, s); \
    a = vaddq_u32(a, b); \
}
#define HH_NEON(a, b, c, d, x, s, ac) { \
    a = vaddq_u32(a, vaddq_u32(H_NEON(b, c, d), vaddq_u32(x, vdupq_n_u32(ac)))); \
    a = ROTATELEFT_NEON(a, s); \
    a = vaddq_u32(a, b); \
}
#define II_NEON(a, b, c, d, x, s, ac) { \
    a = vaddq_u32(a, vaddq_u32(I_NEON(b, c, d), vaddq_u32(x, vdupq_n_u32(ac)))); \
    a = ROTATELEFT_NEON(a, s); \
    a = vaddq_u32(a, b); \
}
Byte *StringProcessNeon(string input, int *n_byte);

void MD5Hash_NEON(vector<std::string>& inputs, uint32x4_t state[4]);
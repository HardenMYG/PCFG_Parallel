#include <iostream>
#include <string>
#include <cstring>
#include <arm_neon.h>
#include <vector>
using namespace std;


// 两条口令并行版本


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



#define F_NEON(x, y, z) vorr_u32(vand_u32(x, y), vand_u32(vmvn_u32(x), z))
#define G_NEON(x, y, z) vorr_u32(vand_u32(x, z), vand_u32(y, vmvn_u32(z)))
#define H_NEON(x, y, z) veor_u32(veor_u32(x, y), z)
#define I_NEON(x, y, z) veor_u32(y, vorr_u32(x, vmvn_u32(z)))

inline uint32x2_t ROTATELEFT_NEON(uint32x2_t num, int n) {
    return vorr_u32(vshl_n_u32(num, n), vshr_n_u32(num, 32 - n));
}

#define FF_NEON(a, b, c, d, x, s, ac) { \
    a = vadd_u32(a, vadd_u32(F_NEON(b, c, d), vadd_u32(x, vdup_n_u32(ac)))); \
    a = ROTATELEFT_NEON(a, s); \
    a = vadd_u32(a, b); \
}
#define GG_NEON(a, b, c, d, x, s, ac) { \
    a = vadd_u32(a, vadd_u32(G_NEON(b, c, d), vadd_u32(x, vdup_n_u32(ac)))); \
    a = ROTATELEFT_NEON(a, s); \
    a = vadd_u32(a, b); \
}
#define HH_NEON(a, b, c, d, x, s, ac) { \
    a = vadd_u32(a, vadd_u32(H_NEON(b, c, d), vadd_u32(x, vdup_n_u32(ac)))); \
    a = ROTATELEFT_NEON(a, s); \
    a = vadd_u32(a, b); \
}
#define II_NEON(a, b, c, d, x, s, ac) { \
    a = vadd_u32(a, vadd_u32(I_NEON(b, c, d), vadd_u32(x, vdup_n_u32(ac)))); \
    a = ROTATELEFT_NEON(a, s); \
    a = vadd_u32(a, b); \
}
Byte *StringProcessNeon(string input, int *n_byte);

void MD5Hash_NEON(vector<std::string>& inputs, uint32x2_t state[4]);
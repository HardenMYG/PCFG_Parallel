// md5_sse.h
#include <xmmintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <iostream>
#include <string>
#include <cstring>
#include <vector> 

using namespace std;


typedef unsigned char Byte;
typedef unsigned int bit32;

// MD5常量定义
#define s11 7
#define s12 12
// ... 其他常量定义保持不变 ...

// SSE版基础函数宏定义
#define F_SSE(x, y, z) _mm_or_si128( \
    _mm_and_si128(x, y), \
    _mm_andnot_si128(x, z) \
)

#define G_SSE(x, y, z) _mm_or_si128( \
    _mm_and_si128(x, z), \
    _mm_andnot_si128(z, y) \
)

#define H_SSE(x, y, z) _mm_xor_si128( \
    _mm_xor_si128(x, y), z \
)

#define I_SSE(x, y, z) _mm_xor_si128( \
    y, _mm_or_si128(x, _mm_andnot_si128(z, _mm_set1_epi32(0xFFFFFFFF))) \
)

// SSE版循环左移
inline __m128i ROTATELEFT_SSE(__m128i num, int n) {
    return _mm_or_si128(
        _mm_slli_epi32(num, n),
        _mm_srli_epi32(num, 32 - n)
    );
}

// SSE版轮函数宏
#define FF_SSE(a, b, c, d, x, s, ac) { \
    a = _mm_add_epi32(a, _mm_add_epi32( \
        F_SSE(b, c, d), \
        _mm_add_epi32(x, _mm_set1_epi32(ac)) \
    )); \
    a = ROTATELEFT_SSE(a, s); \
    a = _mm_add_epi32(a, b); \
}

#define GG_SSE(a, b, c, d, x, s, ac) { \
    a = _mm_add_epi32(a, _mm_add_epi32( \
        G_SSE(b, c, d), \
        _mm_add_epi32(x, _mm_set1_epi32(ac)) \
    )); \
    a = ROTATELEFT_SSE(a, s); \
    a = _mm_add_epi32(a, b); \
}

#define HH_SSE(a, b, c, d, x, s, ac) { \
  a = _mm_add_epi32(a, _mm_add_epi32( \
      H_SSE(b, c, d), \
      _mm_add_epi32(x, _mm_set1_epi32(ac)) \
  )); \
  a = ROTATELEFT_SSE(a, s); \
  a = _mm_add_epi32(a, b); \
}

#define II_SSE(a, b, c, d, x, s, ac) { \
  a = _mm_add_epi32(a, _mm_add_epi32( \
      I_SSE(b, c, d), \
      _mm_add_epi32(x, _mm_set1_epi32(ac)) \
  )); \
  a = ROTATELEFT_SSE(a, s); \
  a = _mm_add_epi32(a, b); \
}

// 函数声明
void MD5Hash_SSE(const vector<std::string>& inputs, __m128i state[4]);
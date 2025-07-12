#include "md5_neon2.h"
#include <iomanip>
#include <assert.h>
#include <chrono>
#include <arm_neon.h>
#include <iostream>
#include <algorithm>
#include <cmath>
using namespace std;
using namespace chrono;

// 两条口令并行版本

/**
 * StringProcess: 将单个输入字符串转换成MD5计算所需的消息数组
 * @param input 输入
 * @param[out] n_byte 用于给调用者传递额外的返回值，即最终Byte数组的长度
 * @return Byte消息数组
 */
Byte *StringProcessNeon(string input, int *n_byte)
{
	// 将输入的字符串转换为Byte为单位的数组
	Byte *blocks = (Byte *)input.c_str();
	int length = input.length();

	// 计算原始消息长度（以比特为单位）
	int bitLength = length * 8;

	// paddingBits: 原始消息需要的padding长度（以bit为单位）
	// 对于给定的消息，将其补齐至length%512==448为止
	// 需要注意的是，即便给定的消息满足length%512==448，也需要再pad 512bits
	int paddingBits = bitLength % 512;
	if (paddingBits > 448)
	{
		paddingBits += 512 - (paddingBits - 448);
	}
	else if (paddingBits < 448)
	{
		paddingBits = 448 - paddingBits;
	}
	else if (paddingBits == 448)
	{
		paddingBits = 512;
	}

	// 原始消息需要的padding长度（以Byte为单位）
	int paddingBytes = paddingBits / 8;
	// 创建最终的字节数组
	// length + paddingBytes + 8:
	// 1. length为原始消息的长度（bits）
	// 2. paddingBytes为原始消息需要的padding长度（Bytes）
	// 3. 在pad到length%512==448之后，需要额外附加64bits的原始消息长度，即8个bytes
	int paddedLength = length + paddingBytes + 8;
	Byte *paddedMessage = new Byte[paddedLength];

	// 复制原始消息
	memcpy(paddedMessage, blocks, length);

	// 添加填充字节。填充时，第一位为1，后面的所有位均为0。
	// 所以第一个byte是0x80
	paddedMessage[length] = 0x80;							 // 添加一个0x80字节
	memset(paddedMessage + length + 1, 0, paddingBytes - 1); // 填充0字节

	// 添加消息长度（64比特，小端格式）
	for (int i = 0; i < 8; ++i)
	{
		// 特别注意此处应当将bitLength转换为uint64_t
		// 这里的length是原始消息的长度
		paddedMessage[length + paddingBytes + i] = ((uint64_t)length * 8 >> (i * 8)) & 0xFF;
	}

	// 验证长度是否满足要求。此时长度应当是512bit的倍数
	int residual = 8 * paddedLength % 512;
	// assert(residual == 0);

	// 在填充+添加长度之后，消息被分为n_blocks个512bit的部分
	*n_byte = paddedLength;
	return paddedMessage;
}


void MD5Hash_NEON(vector<std::string>& inputs, uint32x2_t state[4]){
    const int batch = 2; // 同时处理4个消息

    state[0] = vdup_n_u32(0x67452301);
    state[1] = vdup_n_u32(0xefcdab89);
    state[2] = vdup_n_u32(0x98badcfe);
    state[3] = vdup_n_u32(0x10325476);

        Byte* paddedMessages[batch];
        int lengths[batch];

        // 填充处理
        for (int j = 0; j < batch; ++j) {
            paddedMessages[j] = StringProcessNeon(inputs[j], &lengths[j]);
        }

		int n_blocks = lengths[0] / 64;

        // 加载四个消息块到向量寄存器
		for (int block = 0; block < n_blocks; ++block) {
        uint32x2_t x[16];
        for (int k = 0; k < 16; ++k) {
            uint32_t x_buf[batch] = {0};
            for (int j = 0; j < batch; ++j) {
                x_buf[j] = *reinterpret_cast<uint32_t*>(paddedMessages[j] + block * 64+k * 4);
            }
            x[k] = vld1_u32(x_buf);
        }

        // 初始化状态向量
        uint32x2_t a = state[0], b = state[1], c = state[2], d = state[3];

        /* Round 1 */
		FF_NEON(a, b, c, d, x[0], s11, 0xd76aa478);
		FF_NEON(d, a, b, c, x[1], s12, 0xe8c7b756);
		FF_NEON(c, d, a, b, x[2], s13, 0x242070db);
		FF_NEON(b, c, d, a, x[3], s14, 0xc1bdceee);
		FF_NEON(a, b, c, d, x[4], s11, 0xf57c0faf);
		FF_NEON(d, a, b, c, x[5], s12, 0x4787c62a);
		FF_NEON(c, d, a, b, x[6], s13, 0xa8304613);
		FF_NEON(b, c, d, a, x[7], s14, 0xfd469501);
		FF_NEON(a, b, c, d, x[8], s11, 0x698098d8);
		FF_NEON(d, a, b, c, x[9], s12, 0x8b44f7af);
		FF_NEON(c, d, a, b, x[10], s13, 0xffff5bb1);
		FF_NEON(b, c, d, a, x[11], s14, 0x895cd7be);
		FF_NEON(a, b, c, d, x[12], s11, 0x6b901122);
		FF_NEON(d, a, b, c, x[13], s12, 0xfd987193);
		FF_NEON(c, d, a, b, x[14], s13, 0xa679438e);
		FF_NEON(b, c, d, a, x[15], s14, 0x49b40821);

		/* Round 2 */
		GG_NEON(a, b, c, d, x[1], s21, 0xf61e2562);
		GG_NEON(d, a, b, c, x[6], s22, 0xc040b340);
		GG_NEON(c, d, a, b, x[11], s23, 0x265e5a51);
		GG_NEON(b, c, d, a, x[0], s24, 0xe9b6c7aa);
		GG_NEON(a, b, c, d, x[5], s21, 0xd62f105d);
		GG_NEON(d, a, b, c, x[10], s22, 0x2441453);
		GG_NEON(c, d, a, b, x[15], s23, 0xd8a1e681);
		GG_NEON(b, c, d, a, x[4], s24, 0xe7d3fbc8);
		GG_NEON(a, b, c, d, x[9], s21, 0x21e1cde6);
		GG_NEON(d, a, b, c, x[14], s22, 0xc33707d6);
		GG_NEON(c, d, a, b, x[3], s23, 0xf4d50d87);
		GG_NEON(b, c, d, a, x[8], s24, 0x455a14ed);
		GG_NEON(a, b, c, d, x[13], s21, 0xa9e3e905);
		GG_NEON(d, a, b, c, x[2], s22, 0xfcefa3f8);
		GG_NEON(c, d, a, b, x[7], s23, 0x676f02d9);
		GG_NEON(b, c, d, a, x[12], s24, 0x8d2a4c8a);

		/* Round 3 */
		HH_NEON(a, b, c, d, x[5], s31, 0xfffa3942);
		HH_NEON(d, a, b, c, x[8], s32, 0x8771f681);
		HH_NEON(c, d, a, b, x[11], s33, 0x6d9d6122);
		HH_NEON(b, c, d, a, x[14], s34, 0xfde5380c);
		HH_NEON(a, b, c, d, x[1], s31, 0xa4beea44);
		HH_NEON(d, a, b, c, x[4], s32, 0x4bdecfa9);
		HH_NEON(c, d, a, b, x[7], s33, 0xf6bb4b60);
		HH_NEON(b, c, d, a, x[10], s34, 0xbebfbc70);
		HH_NEON(a, b, c, d, x[13], s31, 0x289b7ec6);
		HH_NEON(d, a, b, c, x[0], s32, 0xeaa127fa);
		HH_NEON(c, d, a, b, x[3], s33, 0xd4ef3085);
		HH_NEON(b, c, d, a, x[6], s34, 0x4881d05);
		HH_NEON(a, b, c, d, x[9], s31, 0xd9d4d039);
		HH_NEON(d, a, b, c, x[12], s32, 0xe6db99e5);
		HH_NEON(c, d, a, b, x[15], s33, 0x1fa27cf8);
		HH_NEON(b, c, d, a, x[2], s34, 0xc4ac5665);

		/* Round 4 */
		II_NEON(a, b, c, d, x[0], s41, 0xf4292244);
		II_NEON(d, a, b, c, x[7], s42, 0x432aff97);
		II_NEON(c, d, a, b, x[14], s43, 0xab9423a7);
		II_NEON(b, c, d, a, x[5], s44, 0xfc93a039);
		II_NEON(a, b, c, d, x[12], s41, 0x655b59c3);
		II_NEON(d, a, b, c, x[3], s42, 0x8f0ccc92);
		II_NEON(c, d, a, b, x[10], s43, 0xffeff47d);
		II_NEON(b, c, d, a, x[1], s44, 0x85845dd1);
		II_NEON(a, b, c, d, x[8], s41, 0x6fa87e4f);
		II_NEON(d, a, b, c, x[15], s42, 0xfe2ce6e0);
		II_NEON(c, d, a, b, x[6], s43, 0xa3014314);
		II_NEON(b, c, d, a, x[13], s44, 0x4e0811a1);
		II_NEON(a, b, c, d, x[4], s41, 0xf7537e82);
		II_NEON(d, a, b, c, x[11], s42, 0xbd3af235);
		II_NEON(c, d, a, b, x[2], s43, 0x2ad7d2bb);
		II_NEON(b, c, d, a, x[9], s44, 0xeb86d391);

        

        state[0] = vadd_u32(state[0], a);
        state[1] = vadd_u32(state[1], b);
        state[2] = vadd_u32(state[2], c);
        state[3] = vadd_u32(state[3], d);
	

    
		}

		// 释放内存
		for (int j = 0; j < batch; ++j) {
			delete[] paddedMessages[j];
		}

		for (int i = 0; i < 4; ++i) {  
			uint8x8_t val_u8 = vreinterpret_u8_u32(state[i]);  
			val_u8 = vrev32_u8(val_u8);  
			state[i] = vreinterpret_u32_u8(val_u8);  
		}  
}
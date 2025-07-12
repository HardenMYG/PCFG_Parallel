#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include "md5_sse.h"
#include <iomanip>
#include <vector>  // 添加vector头文件
using namespace std;
using namespace chrono;

// 编译指令：
// g++ correctness.cpp md5.cpp md5_sse.cpp -o correctness -msse4 -O2

// 打印SSE计算结果
void PrintSSEResult(const __m128i state[4]) {
    alignas(16) uint32_t results[4][4];
    for (int i = 0; i < 4; ++i) {
        _mm_store_si128((__m128i*)results[i], state[i]);
    }

    for (int msg = 0; msg < 4; ++msg) {
        cout << "SSE Hash " << msg+1 << ": ";
        for (int j = 0; j < 4; ++j) {
            cout << hex << setw(8) << setfill('0') << results[j][msg];
        }
        cout << endl;
    }
}

int main() {
    // 测试单个输入

    vector<string> batch_inputs={"","abc","123","!q$%^"}; // 填充4个相同输入

    // 标准MD5计算
    bit32 state[4][4];
    MD5Hash("", state[0]);
    MD5Hash("abc", state[1]);
    MD5Hash("123", state[2]);
    MD5Hash("!q$%^", state[3]);
    
    // SSE MD5计算
    __m128i sse_state[4];
    MD5Hash_SSE(batch_inputs, sse_state);

    // 打印结果
    cout << "Standard MD5: "<<endl;
    for(int i=0;i<4;i++){
        cout << "Hash " << i+1 << ": ";
        for (int i1 = 0; i1 < 4; i1 += 1){
            cout << std::setw(8) << std::setfill('0') << hex << state[i][i1];
        }
        cout << endl;
    }


    cout << "SSE MD5: "<<endl;
    PrintSSEResult(sse_state);

    return 0;
}
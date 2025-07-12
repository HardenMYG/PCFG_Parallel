#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include "md5_neon4.h"
#include <iomanip>
#include<vector>
using namespace std;
using namespace chrono;

// 编译指令如下：
// g++ correctness4.cpp train.cpp guessing.cpp md5_neon4.cpp md5.cpp -o main

void PrintNEONResult(const uint32x4_t state[4]) {
    uint32_t results[4][4];
    vst1q_u32(results[0], state[0]);
    vst1q_u32(results[1], state[1]);
    vst1q_u32(results[2], state[2]);
    vst1q_u32(results[3], state[3]);

    for (int msg = 0; msg < 4; ++msg) {
        cout << "Hash " << msg+1 << ": ";
        for (int j = 0; j < 4; ++j) {
            cout << hex << setw(8) << setfill('0') << results[j][msg];
        }
        cout << endl;
    }
}

// 通过这个函数，你可以验证你实现的SIMD哈希函数的正确性
int main()
{
    bit32 state[4][4];
    MD5Hash("", state[0]);
    MD5Hash("abc", state[1]);
    MD5Hash("123", state[2]);
    MD5Hash("!q$%^", state[3]);
    uint32x4_t neon_state[4];
    vector<string> batch={"","abc","123","!q$%^"};
    MD5Hash_NEON(batch,neon_state);

    cout << "Standard MD5: "<<endl;
    for(int i=0;i<4;i++){
        cout << "Hash " << i+1 << ": ";
        for (int i1 = 0; i1 < 4; i1 += 1){
            cout << std::setw(8) << std::setfill('0') << hex << state[i][i1];
        }
        cout << endl;
    }

    cout << "NEON MD5: "<<endl;
    PrintNEONResult(neon_state);

}


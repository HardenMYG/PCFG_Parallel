#include "PCFG.h"
#include <chrono>
#include <fstream>

#include "md5_neon2.h"
#include "md5.h"
#include <iomanip>
#include <vector>
using namespace std;
using namespace chrono;

//aarch64-linux-gnu-g++ -static -o test -march=armv8.2-a main_neon.cpp train.cpp guessing.cpp md5.cpp md5_neon.cpp -o main_simd.exe -O2
//qemu-aarch64-static ./main_simd.exe

// g++ main_neon2.cpp train.cpp guessing.cpp md5_neon2.cpp md5.cpp -o main -O2
// 两条口令并行版本
int main() {
    double time_hash = 0;  
    double time_guess = 0; 
    double time_train = 0; 
    PriorityQueue q;
    
  
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "System initialized" << endl;
    int curr_num = 0;
    auto start = system_clock::now();
    int history = 0;
    
    // 批量处理缓冲区

    vector<string> batch_buffer;
    uint32x2_t batch_states[4];

    while (!q.priority.empty()) {
        q.PopNext();
        q.total_guesses = q.guesses.size();
        
        if (q.total_guesses - curr_num >= 100000) {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;
            
            int generate_n = 10000000;
            if (history + q.total_guesses > generate_n) {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time: " << time_guess - time_hash << " seconds" << endl;
                cout << "Hash time: " << time_hash << " seconds" << endl;
                cout << "Train time: " << time_train << " seconds" << endl;
                break;
            }
        }

        if (curr_num > 1000000) {
            auto start_hash = system_clock::now();
            
            size_t total = q.guesses.size();
            size_t int_total=(total/2)*2;
            size_t left=total-int_total;
            // 每次处理 4 个口令
            for (size_t i = 0; i < int_total; i += 2) 
            {
                batch_buffer.clear();
                // 收集 4 个口令
                for (int j = 0; j < 2; ++j) 
                {
                    batch_buffer.push_back(q.guesses[i + j]);
                }
                MD5Hash_NEON(batch_buffer, batch_states);
            }

            bit32 state[4];
            for(int i=0;i<left;i++){
                MD5Hash(q.guesses[int_total+i],state);
            }
           

            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }
}
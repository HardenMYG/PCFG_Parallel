
#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <cuda_runtime.h>
using namespace std;
using namespace chrono;

//   nvcc -arch=sm_70 -std=c++11 main_gpu_mul.cu md5.cu train.cu guess_gpu_mul.cu -lcudart -o main_gpu_mul 


//   ./main_gpu_mul

//不进行编译优化
// Guess time: 4.34102 seconds
// Hash time: 6.00593 seconds
// Train time: 62.0473 seconds

//O1
// Guess time: 1.21496 seconds
// Hash time: 2.25465 seconds
// Train time: 13.0445 seconds

//O2
// Guess time:  1.21496 seconds
// Hash time: 2.15669 seconds
// Train time: 12.377 seconds

int main(int argc, char* argv[])
{
    
    // 计时
    auto start_total = high_resolution_clock::now();
    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;

    PriorityQueue q;
    
    auto start_train = high_resolution_clock::now();
    cout << "Starting model training..." << endl;
    q.m.train("input/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = high_resolution_clock::now();
    time_train = duration_cast<duration<double>>(end_train - start_train).count();
    cout << "Model training completed in " << time_train << " seconds" << endl;

    q.init();
    
    cout << "Starting password generation with CUDA..." << endl;
    
    // 使用last_reported跟踪上次报告的猜测数量
    int last_reported = 0;
    
    // 在此处更改实验生成的猜测上限
    int generate_n = 10000000;
    
    // 记录纯猜测时间的起始点
    auto time_guess_start = high_resolution_clock::now();
    
    while (!q.priority.empty())
    {
        // 批量处理：一次处理多个PT
        int batch_size = min(25, (int)q.priority.size());
        q.PopNextBatch_cuda(batch_size);
        
        // 每生成100000个猜测报告一次进度
        if (q.total_guesses - last_reported >= 100000)
        {
            cout << "Guesses generated: " << q.total_guesses << endl;
            last_reported = q.total_guesses;
        }

        // 检查是否达到生成上限
        if (q.total_guesses > generate_n)
        {
            // 计算最后一段猜测时间
            auto time_guess_end = high_resolution_clock::now();
            time_guess += duration_cast<duration<double>>(time_guess_end - time_guess_start).count();
            
            cout << "=== Timing Results ===" << endl;
            cout << "Guess time: " << time_guess << " seconds" << endl;
            cout << "Hash time: " << time_hash << " seconds" << endl;
            cout << "Train time: " << time_train << " seconds" << endl;
            
            auto end_total = high_resolution_clock::now();
            double total_time = duration_cast<duration<double>>(end_total - start_total).count();
            break;
        }
        
        // 当生成的猜测数量达到阈值时执行哈希
        if (q.total_guesses > 1000000)
        {
            // 结束当前段的猜测时间统计
            auto time_guess_end = high_resolution_clock::now();
            time_guess += duration_cast<duration<double>>(time_guess_end - time_guess_start).count();
            
            // 开始哈希时间统计
            auto time_hash_start = high_resolution_clock::now();
            
            bit32 state[4];
            cout<< "Starting hashing for " << q.guesses.size() << " guesses..." << endl;
            for (const string& pw : q.guesses) {    
                MD5Hash(pw, state);
            }
            
            
            
            // 结束哈希时间统计
            auto time_hash_end = high_resolution_clock::now();
            double hash_duration = duration_cast<duration<double>>(time_hash_end - time_hash_start).count();
            time_hash += hash_duration;
            
            // 重置猜测时间起点，准备下一轮
            time_guess_start = high_resolution_clock::now();
           
            // 清空猜测列表释放内存
            q.guesses.clear();
        }
    }
    return 0;
}
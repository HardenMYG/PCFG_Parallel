// 编译指令如下
//mpic++ main_mpi_opt.cpp train.cpp guessing_mpi_opt.cpp md5.cpp -o main_mpi_opt -std=c++11
//mpic++ main_mpi_opt.cpp train.cpp guessing_mpi_opt.cpp md5.cpp -o main_mpi_opt -O1 -std=c++11
//mpic++ main_mpi_opt.cpp train.cpp guessing_mpi_opt.cpp md5.cpp -o main_mpi_opt -O2 -std=c++11


//   mpirun --allow-run-as-root -np 8 ./main_mpi_opt
#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h>
using namespace std;
using namespace chrono;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 使用MPI计时工具
    double mpi_time_start_total = MPI_Wtime();  // 总体开始时间
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 密码猜测的总时长(不包含哈希)
    double time_train = 0; // 模型训练的总时长
    
    PriorityQueue q;
    
    // MPI计时：训练阶段
    double mpi_time_train_start = MPI_Wtime();
    if (rank == 0) {
        cout << "Starting model training..." << endl;
    }
    
    q.m.train("input/Rockyou-singleLined-full.txt");
    q.m.order();
    
    double mpi_time_train_end = MPI_Wtime();
    time_train = mpi_time_train_end - mpi_time_train_start;
    
    if (rank == 0) {
        cout << "Model training completed in " << time_train << " seconds" << endl;
    }

    // 等待所有进程完成训练
    MPI_Barrier(MPI_COMM_WORLD);

    q.init();
    
    if (rank == 0) {
        cout << "Starting password generation with " << size << " MPI processes..." << endl;
    }
    
    // 使用last_reported跟踪上次报告的猜测数量
    int last_reported = 0;
    
    // 在此处更改实验生成的猜测上限
    int generate_n = 10000000;
    
    //Guess和Hash时间
    double mpi_time_start = MPI_Wtime();

    // 记录纯猜测时间的起始点
    double mpi_time_guess_real_start = MPI_Wtime();
    
    while (!q.priority.empty())
    {
        // 批量处理：一次处理size个PT
        q.PopNextBatch_mpi(size);
        
        // 每生成100000个猜测报告一次进度
        if (q.total_guesses - last_reported >= 100000)
        {
            if (rank == 0) {
                cout << "Global guesses generated: " << q.total_guesses << endl;
            }
            last_reported = q.total_guesses;
        }

        // 检查是否达到生成上限
        if (q.total_guesses > generate_n)
        {
            // 计算最后一段猜测时间
            double mpi_time_guess_end = MPI_Wtime();
            double final_guess_duration = mpi_time_guess_end - mpi_time_guess_real_start;
            time_guess += final_guess_duration;
            
            if (rank == 0) {
                cout << "=== MPI Timing Results ===" << endl;
                cout << "Guess time: " << time_guess << " seconds" << endl;
                cout << "Hash time: " << time_hash << " seconds" << endl;
                cout << "Train time: " << time_train << " seconds" << endl;
                
                double total_time = MPI_Wtime() - mpi_time_start_total;
                cout << "Total execution time: " << total_time << " seconds" << endl;
                cout << "Time breakdown: " << endl;
                cout << "  - Training: " << (time_train/total_time)*100 << "%" << endl;
                cout << "  - Guessing: " << (time_guess/total_time)*100 << "%" << endl;
                cout << "  - Hashing: " << (time_hash/total_time)*100 << "%" << endl;
            }
            break;
        }
        
        // 当生成的猜测数量达到阈值时执行哈希
        if (q.total_guesses > 1000000)
        {
            // 结束当前段的猜测时间统计
            double mpi_time_guess_end = MPI_Wtime();
            double guess_duration = mpi_time_guess_end - mpi_time_guess_real_start;
            time_guess += guess_duration;
            
            // 开始哈希时间统计
            double mpi_time_hash_start = MPI_Wtime();
            
            bit32 state[4];
            for (string pw : q.guesses)
            {
                // TODO：对于SIMD实验，将这里替换成你的SIMD MD5函数
                MD5Hash(pw, state);
            }

            // 结束哈希时间统计
            double mpi_time_hash_end = MPI_Wtime();
            double hash_duration = mpi_time_hash_end - mpi_time_hash_start;
            time_hash += hash_duration;
            
            // 重置猜测时间起点，准备下一轮
            mpi_time_guess_real_start = MPI_Wtime();
            
            if (rank == 0) {
                cout << "Hash batch completed in " << hash_duration << " seconds" << endl;
            }

            // 清空猜测列表释放内存
            q.guesses.clear();
        }
    }
    
    // 最终的MPI计时统计
    double mpi_time_end_total = MPI_Wtime();
    
    if (rank == 0) {

        cout << "\n=== Final MPI Timing Summary ===" << endl;
        cout << "Number of MPI processes: " << size << endl;
        cout << "Guess And Hash Time: " << mpi_time_end_total-mpi_time_start << " seconds" << endl;

    }
    MPI_Finalize();
    return 0;
}

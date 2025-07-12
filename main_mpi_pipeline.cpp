// main_mpi_pipeline.cpp
// 编译指令如下
//mpic++ main_mpi_pipeline.cpp train.cpp guessing_mpi_pipeline.cpp md5.cpp -o main_mpi_pipeline -std=c++11
//mpic++ main_mpi_pipeline.cpp train.cpp guessing_mpi_pipeline.cpp md5.cpp -o main_mpi_pipeline -O1 -std=c++11
//mpic++ main_mpi_pipeline.cpp train.cpp guessing_mpi_pipeline.cpp md5.cpp -o main_mpi_pipeline -O2 -std=c++11


//   mpirun --allow-run-as-root -np 2 ./main_mpi_pipeline
#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h>
#include <algorithm> // 包含min函数

using namespace std;
using namespace chrono;

int main(int argc, char* argv[]) {
    // 初始化MPI环境
    MPI_Init(&argc, &argv);
    
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // 时间点变量
    time_point<high_resolution_clock> train_end_time;
    time_point<high_resolution_clock> process_start_time;
    time_point<high_resolution_clock> process_end_time;
    
    double train_time = 0.0;
    double total_time = 0.0;
    
    // 确保有两个进程
    if (world_size < 2) {
        cerr << "This application requires at least 2 MPI processes." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // 定义最大口令数
    const int MAX_GUESSES = 10000000;

    if (world_rank == 0) { // 口令生成进程
        PriorityQueue q;
        auto start_train = high_resolution_clock::now();
        q.m.train("./input/Rockyou-singleLined-full.txt");
        q.m.order();
        train_end_time = high_resolution_clock::now();
        train_time = duration_cast<duration<double>>(train_end_time - start_train).count();
        
        // 记录训练结束后的开始时间
        process_start_time = high_resolution_clock::now();
        
        cout<<"START INIT"<<endl;
        q.init();
        cout<<"END INIT"<<endl;
        int total_guesses = 0;
        
        while (!q.priority.empty() && total_guesses < MAX_GUESSES) {
            cout<<"START POP"<<endl;
            q.PopNext();
            cout<<"END POP"<<endl;
            
            // 检查是否达到或超过最大口令数
            if (total_guesses + q.guesses.size() >= MAX_GUESSES) {
                total_guesses = MAX_GUESSES;
                break;  // 立即退出循环
            }
            // 正常发送整批口令
            else if (q.guesses.size() >= 1000000) {
                int batch_size = q.guesses.size();
                MPI_Send(&batch_size, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
                
                for (string& pw : q.guesses) {
                    int len = pw.size();
                    MPI_Send(&len, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
                    MPI_Send(pw.c_str(), len, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                }
                
                total_guesses += batch_size;
                q.guesses.clear();
            }
        }
        
        // 记录整个处理结束时间
        process_end_time = high_resolution_clock::now();
        total_time = duration_cast<duration<double>>(process_end_time - process_start_time).count();
        
        // 输出统计信息
        cout << fixed << setprecision(4);
        cout << "Train time: " << train_time << " seconds" << endl;
        cout << "Total execution time after training: " << total_time << " seconds" << endl;
        cout << "Total guesses generated: " << total_guesses << endl;
        
        // 发送终止信号
        int end_signal = -1;
        MPI_Send(&end_signal, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } 
    else if (world_rank == 1) { // 哈希计算进程
        int total_hashes = 0;
        
        while (true) {
            int batch_size;
            MPI_Recv(&batch_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // 检查是否收到结束信号
            if (batch_size == -1) break;
            
            for (int i = 0; i < batch_size; i++) {
                int len;
                MPI_Recv(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                char* buffer = new char[len + 1];
                MPI_Recv(buffer, len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                buffer[len] = '\0';
                string pw(buffer);
                delete[] buffer;
                
                // 计算哈希
                bit32 state[4];
                MD5Hash(pw, state);
            }
            total_hashes += batch_size;
        }
        cout << "Total hashes computed: " << total_hashes << endl;
    }

    MPI_Finalize();
    return 0;
}
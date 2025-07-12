#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"

#include <iomanip>
using namespace std;
using namespace chrono;



// g++ main_openmp.cpp  train.cpp guessing_openmp.cpp md5.cpp  -o main_openmp -fopenmp
// g++ main_openmp.cpp  train.cpp guessing_openmp.cpp md5.cpp  -o main_openmp -O1 -fopenmp
// g++ main_openmp.cpp  train.cpp guessing_openmp.cpp md5.cpp  -o main_openmp -O2 -fopenmp -g

//O
//Guess time:7.42963seconds
//Hash time:12.1218seconds


//O1

//Guess time:0.396461seconds
//Hash time:1.69082seconds

// Guess time:0.407213seconds
// Hash time:1.64793seconds

//O2


//Guess time:0.397428seconds
//Hash time:1.68725seconds


//以下结果均是平均所得(可能去除了个别奇异值或者由于口令数量过大只进行了一次)

////2   
//0
//1:Guess time:7.41376seconds

//10:Guess time:58.4977seconds
//100:对于不编译优化的10亿条口令耗时猜测会接近一个小时，这里就不进行测试了。

//1
//0.410
//1: 0.439711 0.398221  0.421991  0.395427  0.395643 0.502261
//10 4.15
//100:// Guess time:143.255seconds
// Hash time:186.876seconds


//2
//1:0.380
//  0.414389 0.386097  0.386393 0.36825 0.346645
//10:4.03
//100:Guess time:121.996seconds
// Hash time:178.756seconds


////4   
//0
//1:7.39861
//10:56.3821
//100:对于不编译优化的10亿条口令耗时猜测会接近一个小时，这里就不进行测试了。

//1
//0.379
//1:0.405258  0.387795   0.368856   0.366244  0.354674   0.389505
//10:3.5352
//100:Guess time:122.205seconds
// Hash time:174.879seconds

//2
//0.359
//1:0.373987  0.363645   0.351409   0.33793  0.368227  
//10: 3.12506
//100:Guess time:120.384seconds
// Hash time:184.003seconds


////8   
//0
//1:7.05867
//10:55.5222
//100:对于不编译优化的10亿条口令耗时猜测会接近一个小时，这里就不进行测试了。

// Hash time:12.0275seconds

//1
//1:0.357
//10:3.10
//100:Guess time:120.025seconds
// Hash time:190.127seconds

// Guess time:0.358286seconds
// Hash time:1.7883seconds

// Guess time:0.338966seconds
// Hash time:1.76235seconds

// Guess time:0.35255seconds
// Hash time:1.72479seconds

// Guess time:0.363543seconds
// Hash time:1.76545seconds

// Guess time:0.370208seconds
// Hash time:1.74635seconds



//2
//1:0.301
//10:3.02
//100:Guess time:112.441seconds
// Hash time:173.801seconds

// Guess time:0.29827seconds
// Hash time:1.64981seconds

// Guess time:0.338163seconds 忽略
// Hash time:1.62553seconds

// Guess time:0.294212seconds
// Hash time:1.69676seconds

// Guess time:0.309421seconds
// Hash time:1.63718seconds

// Guess time:0.292767seconds
// Hash time:1.6966seconds

// Guess time:0.364024seconds 忽略
// Hash time:1.8093seconds

// Guess time:0.30333seconds
// Hash time:1.68748seconds

// Guess time:0.309546seconds
// Hash time:1.71063seconds
int main()
{
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("./input/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "here" << endl;
    int curr_num = 0;
    auto start = system_clock::now();
    // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    int history = 0;
    // std::ofstream a("./files/results.txt");

    

    while (!q.priority.empty())
    {

        q.PopNext();

        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " <<history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            // 在此处更改实验生成的猜测上限
            int generate_n=100000000;
            if (history + q.total_guesses > 100000000)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << "seconds"<< endl;
                cout << "Hash time:" << time_hash << "seconds"<<endl;
                cout << "Train time:" << time_train <<"seconds"<<endl;

                break;
            }
        }
        // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
        // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
        if (curr_num > 1000000)
        {
            auto start_hash = system_clock::now();

            
            bit32 state[4];
            for(int i=0;i<num_thread;i++){
            for (string pw : q.threadargs[i].local_guesses)
            {
                // TODO：对于SIMD实验，将这里替换成你的SIMD MD5函数
                MD5Hash(pw, state);
            }
        }

            // 在这里对哈希所需的总时长进行计算
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            // 记录已经生成的口令总数
            history += curr_num;
            curr_num = 0;
            for (int i = 0; i < num_thread; i++) {
                q.threadargs[i].local_guesses.clear();
            }
            q.total_guesses=0;
        }
    }
}

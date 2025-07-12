#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"

#include <iomanip>
using namespace std;
using namespace chrono;




// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2

// g++ main_pthread.cpp  train.cpp guessing_pthread.cpp md5.cpp  -o main_pthread -lpthread
// g++ main_pthread.cpp  train.cpp guessing_pthread.cpp md5.cpp  -o main_pthread -O1 -lpthread
// g++ main_pthread.cpp  train.cpp guessing_pthread.cpp md5.cpp  -o main_pthread -O2 -lpthread -g

//以下结果均是平均所得(可能去除了个别奇异值或者由于口令数量过大只进行了一次)

////2   threshold:2500+-
//0
//1:7.30201
//10:59.6198
//100:对于不编译优化的10亿条口令耗时猜测会接近一个小时，这里就不进行测试了。

//1
//1:0.532
//10:4.652 
//100:133.828

//2
//1: 0.454
//10:4.346
//100:123.348


////4   threshold:4000+-(O2编译优化下)
//0
//1:7.239
//10:58.294
//100:对于不编译优化的10亿条口令耗时猜测会接近一个小时，这里就不进行测试了。

//1
//1:0.445
//10:4.442
//100:133.035

//2
//1:0.402  
//10: 4.263
//100:120.366


////8   threshold:10000+-(O2编译优化下)
//0
//1:7.19787
//10:58.1214
//100:对于不编译优化的10亿条口令耗时猜测会接近一个小时，这里就不进行测试了。

// Hash time:12.0386seconds

// Hash time:11.9958seconds

// Hash time:12.1941seconds

// Hash time:12.1192seconds

// Hash time:12.1054seconds

//1
//1:0.438
//10: 3.954
//100:
// Guess time:132.025seconds

// Guess time:0.442688seconds
// Hash time:1.81171seconds

// Guess time:0.475167seconds
// Hash time:1.76024seconds

// Guess time:0.403813seconds 忽略
// Hash time:1.68999seconds

// Guess time:0.45391seconds
// Hash time:1.76456seconds

// Guess time:0.436497seconds
// Hash time:1.74575seconds

// Guess time:0.423001seconds
// Hash time:1.67343seconds

// Guess time:0.451294seconds
// Hash time:1.77103seconds

// Guess time:0.414736seconds
// Hash time:1.74327seconds

//2
//1: 0.397
//10:3.725
//100:123.357 Guess time:116.744seconds

// Guess time:0.400991seconds
// Hash time:1.65641seconds

// Guess time:0.391607seconds
// Hash time:1.57083seconds


// Guess time:0.417257seconds
// Hash time:1.71492seconds

// Guess time:0.406198seconds
// Hash time:1.64338seconds

// Guess time:0.399015seconds
// Hash time:1.67546seconds


// Guess time:0.359605seconds 忽略
// Hash time:1.6988seconds

// Guess time:0.393011seconds
// Hash time:1.71319seconds

// Guess time:0.372409seconds
// Hash time:1.60583seconds


// bash test.sh 1 1

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

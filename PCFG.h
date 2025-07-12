#include <string>
#include <iostream>
#include <unordered_map>
#include <queue>
#include <omp.h>
#include <vector>
#include <future>
#include <algorithm>
#include <pthread.h>

// #include <chrono>   
// using namespace chrono;
using namespace std;


#define num_thread 8

class segment
{
public:
    int type; // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
    int length; // 长度，例如S6的长度就是6
    segment(int type, int length)
    {
        this->type = type;
        this->length = length;
    };

    // 打印相关信息
    void PrintSeg();

    // 按照概率降序排列的value。例如，123是D3的一个具体value，其概率在D3的所有value中排名第三，那么其位置就是ordered_values[2]
    vector<string> ordered_values;

    // 按照概率降序排列的频数（概率）
    vector<int> ordered_freqs;

    // total_freq作为分母，用于计算每个value的概率
    int total_freq = 0;

    // 未排序的value，其中int就是对应的id
    unordered_map<string, int> values;

    // 根据id，在freqs中查找/修改一个value的频数
    unordered_map<int, int> freqs;


    void insert(string value);
    void order();
    void PrintValues();
};

class PT
{
public:
    // 例如，L6D1的content大小为2，content[0]为L6，content[1]为D1
    vector<segment> content;

    // pivot值，参见PCFG的原理
    int pivot = 0;
    void insert(segment seg);
    void PrintPT();

    // 导出新的PT
    vector<PT> NewPTs();

    // 记录当前每个segment（除了最后一个）对应的value，在模型中的下标
    vector<int> curr_indices;

    // 记录当前每个segment（除了最后一个）对应的value，在模型中的最大下标（即最大可以是max_indices[x]-1）
    vector<int> max_indices;
    // void init();
    float preterm_prob;
    float prob;
};

class model
{
public:
    // 对于PT/LDS而言，序号是递增的
    // 训练时每遇到一个新的PT/LDS，就获取一个新的序号，并且当前序号递增1
    int preterm_id = -1;
    int letters_id = -1;
    int digits_id = -1;
    int symbols_id = -1;
    int GetNextPretermID()
    {
        preterm_id++;
        return preterm_id;
    };
    int GetNextLettersID()
    {
        letters_id++;
        return letters_id;
    };
    int GetNextDigitsID()
    {
        digits_id++;
        return digits_id;
    };
    int GetNextSymbolsID()
    {
        symbols_id++;
        return symbols_id;
    };

    // C++上机和数据结构实验中，一般不允许使用stl
    // 这就导致大家对stl不甚熟悉。现在是时候体会stl的便捷之处了
    // unordered_map: 无序映射
    int total_preterm = 0;
    vector<PT> preterminals;
    int FindPT(PT pt);

    vector<segment> letters;
    vector<segment> digits;
    vector<segment> symbols;
    int FindLetter(segment seg);
    int FindDigit(segment seg);
    int FindSymbol(segment seg);

    unordered_map<int, int> preterm_freq;
    unordered_map<int, int> letters_freq;
    unordered_map<int, int> digits_freq;
    unordered_map<int, int> symbols_freq;

    vector<PT> ordered_pts;

    // 给定一个训练集，对模型进行训练
    void train(string train_path);

    // 对已经训练的模型进行保存
    void store(string store_path);

    // 从现有的模型文件中加载模型
    void load(string load_path);

    // 对一个给定的口令进行切分
    void parse(string pw);

    void order();

    // 打印模型
    void print();
};


struct ThreadArgs {
    segment* seg;          // 最后一个segment的指针
    string guess;         // 已实例化的前缀
    int start;         // 起始索引
    int end;           // 结束索引
    vector<string> local_guesses; 

};

struct PTData {
    std::string prefix;
    std::vector<std::string> values;
    int pt_index;
};


// 异步GPU任务数据结构
struct AsyncGPUData {
    cudaStream_t stream;
    char *d_all_prefixes = nullptr, *d_all_values = nullptr, *d_output = nullptr;
    int *d_prefix_offsets = nullptr, *d_prefix_lengths = nullptr;
    int *d_value_offsets = nullptr, *d_value_lengths = nullptr;
    int *d_pt_starts = nullptr, *d_pt_counts = nullptr, *d_output_offsets = nullptr;
    char *h_all_prefixes = nullptr, *h_all_values = nullptr, *h_output = nullptr;
    vector<int> output_offsets;
    int total_passwords = 0;
    bool initialized = false;
    
    AsyncGPUData() {
        cudaStreamCreate(&stream);
        initialized = true;
    }
    
    ~AsyncGPUData() {
        if (!initialized) return;
        
        // 确保所有操作完成
        cudaStreamSynchronize(stream);
        
        // 释放设备内存
        if (d_all_prefixes) cudaFreeAsync(d_all_prefixes, stream);
        if (d_all_values) cudaFreeAsync(d_all_values, stream);
        if (d_output) cudaFreeAsync(d_output, stream);
        if (d_prefix_offsets) cudaFreeAsync(d_prefix_offsets, stream);
        if (d_prefix_lengths) cudaFreeAsync(d_prefix_lengths, stream);
        if (d_value_offsets) cudaFreeAsync(d_value_offsets, stream);
        if (d_value_lengths) cudaFreeAsync(d_value_lengths, stream);
        if (d_pt_starts) cudaFreeAsync(d_pt_starts, stream);
        if (d_pt_counts) cudaFreeAsync(d_pt_counts, stream);
        if (d_output_offsets) cudaFreeAsync(d_output_offsets, stream);
        
        // 释放主机内存
        if (h_all_prefixes) delete[] h_all_prefixes;
        if (h_all_values) delete[] h_all_values;
        if (h_output) delete[] h_output;
        
        // 销毁流
        cudaStreamDestroy(stream);
        initialized = false;
    }
};

// 优先队列，用于按照概率降序生成口令猜测
// 实际上，这个class负责队列维护、口令生成、结果存储的全部过程
class PriorityQueue
{
public:
    // 用vector实现的priority queue
    vector<PT> priority;

    // 模型作为成员，辅助猜测生成
    model m;

    // 计算一个pt的概率
    void CalProb(PT &pt);

    // 优先队列的初始化
    void init();

    // 对优先队列的一个PT，生成所有guesses
    void Generate(PT pt);

    // 将优先队列最前面的一个PT
    void PopNext();
    int total_guesses = 0;
    vector<string> guesses;

    const int threshold=20000;  //8:10500   4: 4000     2:2500
    pthread_t threads[num_thread];
    ThreadArgs threadargs[num_thread];

	void Generate_mpi(PT pt);
    void PopNextBatch_mpi(int batch_size);
    void ProcessPTBatch_mpi(std::vector<PT>& pt_batch);
    void GenerateSinglePT_mpi(PT& pt, std::vector<std::string>& pt_guesses);
    
    // 添加双缓冲和异步支持
    std::vector<std::string> buffer[2];  // 双缓冲区
    int current_buffer = 0;              // 当前使用的缓冲区索引
    std::future<void> hash_future;       // 异步哈希任务
    bool hash_in_progress = false;       // 标记哈希是否进行中
    
    // 添加哈希函数
    void HashBuffer(std::vector<std::string>& buffer);

    
    void PopNextBatch_cuda(int batch_size);
    // 批量GPU处理函数
    void GenerateBatchOnGPU(const std::vector<PTData>& pt_data_batch, int total_passwords);



    void PopNextBatchCuda(int batch_size);
    void ProcessGPUResults(AsyncGPUData& gpu_data);
    void GenerateBatchOnGPU_Async(const vector<PTData>& pt_data_batch, 
                                           int total_passwords, 
                                           AsyncGPUData& gpu_data);

};

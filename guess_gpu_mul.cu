#include "PCFG.h"
#include <chrono>
#include <queue>
#include <condition_variable>
#include <cstring>
#include <unistd.h>
#include <sstream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
using namespace std;


void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}


// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}

// 批量处理CUDA内核：一次处理多个PT的所有密码生成
__global__ void generate_batch_passwords_kernel(
    char* d_all_prefixes, int* d_prefix_offsets, int* d_prefix_lengths,
    char* d_all_values, int* d_value_offsets, int* d_value_lengths,
    int* d_pt_starts, int* d_pt_counts, int num_pts,
    char* d_output, int* d_output_offsets) {
    
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 找到这个线程对应的PT和其中的密码索引
    int pt_idx = 0;
    int local_password_idx = global_idx;
    
    // 找到global_idx属于哪个PT
    while (pt_idx < num_pts && local_password_idx >= d_pt_counts[pt_idx]) {
        local_password_idx -= d_pt_counts[pt_idx];
        pt_idx++;
    }
    
    if (pt_idx >= num_pts) return; // 超出范围
    
    // 计算实际的值索引
    int value_idx = d_pt_starts[pt_idx] + local_password_idx;
    
    // 获取前缀信息
    char* prefix_ptr = d_all_prefixes + d_prefix_offsets[pt_idx];
    int prefix_len = d_prefix_lengths[pt_idx];
    
    // 获取值信息
    char* value_ptr = d_all_values + d_value_offsets[value_idx];
    int value_len = d_value_lengths[value_idx];
    
    // 获取输出位置
    char* output_ptr = d_output + d_output_offsets[global_idx];
    
    // 生成密码：前缀 + 值
    for (int i = 0; i < prefix_len; i++) {
        output_ptr[i] = prefix_ptr[i];
    }
    for (int i = 0; i < value_len; i++) {
        output_ptr[prefix_len + i] = value_ptr[i];
    }
    output_ptr[prefix_len + value_len] = '\0';
}

// 真正的批量GPU处理函数
void PriorityQueue::PopNextBatch_cuda(int batch_size)
{
    int actual_batch_size = min(batch_size, (int)priority.size());
    if (actual_batch_size == 0) return;
    
    // 预处理批次数据
    vector<PTData> pt_data_batch;
    vector<PT> all_new_pts;
    int total_passwords = 0;
    
    // 收集所有PT的数据
    for (int batch_idx = 0; batch_idx < actual_batch_size; batch_idx++) {
        PT& pt = priority[batch_idx];
        CalProb(pt);
        
        PTData pt_data;
        pt_data.pt_index = batch_idx;
        
        segment* a = nullptr;
        
        if (pt.content.size() == 1) {
            if (pt.content[0].type == 1) {
                a = &m.letters[m.FindLetter(pt.content[0])];
            } else if (pt.content[0].type == 2) {
                a = &m.digits[m.FindDigit(pt.content[0])];
            } else if (pt.content[0].type == 3) {
                a = &m.symbols[m.FindSymbol(pt.content[0])];
            }
            pt_data.prefix = "";
            if (a) pt_data.values = a->ordered_values;
        } else {
            // 构建前缀
            int seg_idx = 0;
            for (int idx : pt.curr_indices) {
                if (seg_idx == pt.content.size() - 1) break;
                if (pt.content[seg_idx].type == 1) {
                    pt_data.prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
                } else if (pt.content[seg_idx].type == 2) {
                    pt_data.prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
                } else if (pt.content[seg_idx].type == 3) {
                    pt_data.prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
                }
                seg_idx++;
            }
            
            // 获取最后一个segment的值
            int last_idx = pt.content.size() - 1;
            if (pt.content[last_idx].type == 1) {
                a = &m.letters[m.FindLetter(pt.content[last_idx])];
            } else if (pt.content[last_idx].type == 2) {
                a = &m.digits[m.FindDigit(pt.content[last_idx])];
            } else if (pt.content[last_idx].type == 3) {
                a = &m.symbols[m.FindSymbol(pt.content[last_idx])];
            }
            if (a) pt_data.values = a->ordered_values;
        }
        
        // 小批量用CPU处理
        if (!pt_data.values.empty() && pt_data.values.size() < threshold) {
            for (const string& value : pt_data.values) {
                guesses.push_back(pt_data.prefix + value);
            }
            total_guesses += pt_data.values.size();
        } else if (!pt_data.values.empty()) {
            pt_data_batch.push_back(pt_data);
            total_passwords += pt_data.values.size();
        }
        
        // 收集新PT
        vector<PT> new_pts = pt.NewPTs();
        for (PT& new_pt : new_pts) {
            CalProb(new_pt);
            all_new_pts.push_back(new_pt);
        }
    }
    
    // GPU批量处理
    if (!pt_data_batch.empty() && total_passwords > 0) {
       GenerateBatchOnGPU(pt_data_batch, total_passwords);
    }
    
    // 移除已处理的PT
    priority.erase(priority.begin(), priority.begin() + actual_batch_size);
    
    // 批量插入新PT
    if (!all_new_pts.empty()) {
        sort(all_new_pts.begin(), all_new_pts.end(), 
             [](const PT& a, const PT& b) { return a.prob > b.prob; });
        
        vector<PT> new_priority;
        new_priority.reserve(priority.size() + all_new_pts.size());
        
        size_t i = 0, j = 0;
        while (i < priority.size() && j < all_new_pts.size()) {
            if (priority[i].prob >= all_new_pts[j].prob) {
                new_priority.push_back(priority[i++]);
            } else {
                new_priority.push_back(all_new_pts[j++]);
            }
        }
        
        while (i < priority.size()) new_priority.push_back(priority[i++]);
        while (j < all_new_pts.size()) new_priority.push_back(all_new_pts[j++]);
        
        priority = move(new_priority);
    }
}

// GPU批量处理核心函数
void PriorityQueue::GenerateBatchOnGPU(const vector<PTData>& pt_data_batch, int total_passwords) {
    
    // 计算内存需求
    size_t total_prefix_size = 0;
    size_t total_values_size = 0;
    size_t total_output_size = 0;
    
    vector<int> prefix_lengths;
    vector<int> prefix_offsets;
    vector<int> pt_counts;
    vector<int> pt_starts;
    vector<int> value_lengths;
    vector<int> value_offsets;
    vector<int> output_offsets;
    
    int current_prefix_offset = 0;
    int current_value_offset = 0;
    int current_output_offset = 0;
    int current_password_start = 0;
    
    for (const auto& pt_data : pt_data_batch) {
        // 前缀信息
        prefix_lengths.push_back(pt_data.prefix.length());
        prefix_offsets.push_back(current_prefix_offset);
        total_prefix_size += pt_data.prefix.length();
        current_prefix_offset += pt_data.prefix.length();
        
        // PT信息
        pt_counts.push_back(pt_data.values.size());
        pt_starts.push_back(current_password_start);
        current_password_start += pt_data.values.size();
        
        // 值信息
        for (const string& value : pt_data.values) {
            value_lengths.push_back(value.length());
            value_offsets.push_back(current_value_offset);
            total_values_size += value.length();
            current_value_offset += value.length();
            
            // 输出信息
            output_offsets.push_back(current_output_offset);
            int password_len = pt_data.prefix.length() + value.length() + 1; // +1 for null
            total_output_size += password_len;
            current_output_offset += password_len;
        }
    }
    
    // 分配和填充主机内存
    char* h_all_prefixes = new char[total_prefix_size];
    char* h_all_values = new char[total_values_size];
    char* h_output = new char[total_output_size];
    
    current_prefix_offset = 0;
    current_value_offset = 0;
    
    for (const auto& pt_data : pt_data_batch) {
        memcpy(h_all_prefixes + current_prefix_offset, pt_data.prefix.c_str(), pt_data.prefix.length());
        current_prefix_offset += pt_data.prefix.length();
        
        for (const string& value : pt_data.values) {
            memcpy(h_all_values + current_value_offset, value.c_str(), value.length());
            current_value_offset += value.length();
        }
    }
    
    // GPU内存分配
    char *d_all_prefixes, *d_all_values, *d_output;
    int *d_prefix_offsets, *d_prefix_lengths, *d_value_offsets, *d_value_lengths;
    int *d_pt_starts, *d_pt_counts, *d_output_offsets;
    
    cudaMalloc(&d_all_prefixes, total_prefix_size);
    cudaMalloc(&d_all_values, total_values_size);
    cudaMalloc(&d_output, total_output_size);
    cudaMalloc(&d_prefix_offsets, pt_data_batch.size() * sizeof(int));
    cudaMalloc(&d_prefix_lengths, pt_data_batch.size() * sizeof(int));
    cudaMalloc(&d_value_offsets, total_passwords * sizeof(int));
    cudaMalloc(&d_value_lengths, total_passwords * sizeof(int));
    cudaMalloc(&d_pt_starts, pt_data_batch.size() * sizeof(int));
    cudaMalloc(&d_pt_counts, pt_data_batch.size() * sizeof(int));
    cudaMalloc(&d_output_offsets, total_passwords * sizeof(int));
    
    // 数据传输到GPU
    cudaMemcpy(d_all_prefixes, h_all_prefixes, total_prefix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_all_values, h_all_values, total_values_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix_offsets, prefix_offsets.data(), pt_data_batch.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix_lengths, prefix_lengths.data(), pt_data_batch.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value_offsets, value_offsets.data(), total_passwords * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value_lengths, value_lengths.data(), total_passwords * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pt_starts, pt_starts.data(), pt_data_batch.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pt_counts, pt_counts.data(), pt_data_batch.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_offsets, output_offsets.data(), total_passwords * sizeof(int), cudaMemcpyHostToDevice);
    
    // 启动GPU内核
    int blockSize = 256;
    int gridSize = (total_passwords + blockSize - 1) / blockSize;
    generate_batch_passwords_kernel<<<gridSize, blockSize>>>(
        d_all_prefixes, d_prefix_offsets, d_prefix_lengths,
        d_all_values, d_value_offsets, d_value_lengths,
        d_pt_starts, d_pt_counts, pt_data_batch.size(),
        d_output, d_output_offsets);
    
    cudaDeviceSynchronize();
    
    // 复制结果回主机
    cudaMemcpy(h_output, d_output, total_output_size, cudaMemcpyDeviceToHost);
    
    // 添加到结果
    for (int i = 0; i < total_passwords; i++) {
        guesses.push_back(string(h_output + output_offsets[i]));
    }
    total_guesses += total_passwords;
    
    // 清理内存
    delete[] h_all_prefixes;
    delete[] h_all_values;
    delete[] h_output;
    cudaFree(d_all_prefixes);
    cudaFree(d_all_values);
    cudaFree(d_output);
    cudaFree(d_prefix_offsets);
    cudaFree(d_prefix_lengths);
    cudaFree(d_value_offsets);
    cudaFree(d_value_lengths);
    cudaFree(d_pt_starts);
    cudaFree(d_pt_counts);
    cudaFree(d_output_offsets);
}
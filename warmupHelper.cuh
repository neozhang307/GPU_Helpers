#include <tuple>
#include <cuda.h>
#include <cooperative_groups.h>
// Simple way to change from traditional launch to cooperative launch
// Simple way to do warmup
// Simple way to do iterative kernel launch.
// Base class with virtual function of launch and warmup
#pragma once 

class WarmupHelper
{
private:
    /* data */
    double warmupaim = 350; // 350ms
    int warmupiteration = 1000;

public:
    WarmupHelper(double warmupaim = 350, int warmupiteration = 1000);
    ~WarmupHelper();

    template <typename Function, typename... Args>
    void warmup(Function func, Args &&...args);
};

WarmupHelper::WarmupHelper(double givenwarmupsec, int givenwarmupiteration)
{
    this->warmupaim = givenwarmupsec;
    this->warmupiteration = givenwarmupiteration;
}

WarmupHelper::~WarmupHelper()
{
}
template < typename Function, typename... Args>
void WarmupHelper::warmup(Function func, Args &&...args)
{
    cudaEvent_t warstart, warmstop;
    cudaEventCreate(&warstart);
    cudaEventCreate(&warmstop);
    cudaEventRecord(warstart, 0);
    for (int i = 0; i < warmupiteration; i++)
    {
        func(std::forward<Args>(args)...);
    }
    // launch(std::forward<Args>(args)...);
    cudaEventRecord(warmstop, 0);
    cudaEventSynchronize(warmstop);
    float warmelapsedTime;
    cudaEventElapsedTime(&warmelapsedTime, warstart, warmstop);
    int nowwarmup = warmelapsedTime;
    int nowiter = (warmupaim + nowwarmup - 1) / nowwarmup;
    for (int out = 0; out < nowiter; out++)
    {
        for (int i = 0; i < warmupiteration; i++)
        {
            // launch(std::forward<Args>(args)...);
            func(std::forward<Args>(args)...);
        }
    }
    cudaEventDestroy(warstart);
    cudaEventDestroy(warmstop);
}

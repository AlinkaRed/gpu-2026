#include "gelu_omp.h"
#include "omp.h"
#include <cmath>
#include <math.h>

using namespace std;

namespace
{

    inline float fast_tanh(float z)
    {
        return 1.0f - 2.0f / (exp(2.0f * z) + 1.0f);
    }

    float kSqrt2OverPi = static_cast<float>(sqrt(2.0 / M_PI));

} // namespace

std::vector<float> GeluOMP(const std::vector<float> &input)
{
    const size_t n = input.size();
    vector<float> v(n);

    const float *__restrict in = input.data();
    float *__restrict out = v.data();

#pragma omp parallel for simd
    for (long long i = 0; i < static_cast<long long>(n); i++)
    {
        const float x = in[i];
        out[i] = 0.5f * x * (1.0f + fast_tanh(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
    }

    return v;
}
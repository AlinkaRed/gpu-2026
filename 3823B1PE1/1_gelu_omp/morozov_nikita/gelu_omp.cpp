#include "gelu_omp.h"

#include <omp.h>
#include <cmath>

static float Gelu(float num) 
{
    constexpr float SQRT_2_DIV_PI = 0.7978845608028654f;
    constexpr float GELU_CONST = 0.044715f;

    float gelu_tanh = std::tanh(SQRT_2_DIV_PI * (num + (GELU_CONST * num * num * num)));

    return (0.5f * num) * (1 + gelu_tanh);
}

std::vector<float> GeluOMP(const std::vector<float>& input)
{
    std::vector<float> res(input.size(), 0.0f);

    #pragma omp parallel for simd schedule(static) default(none) shared(res, input)
    for(size_t i = 0; i < input.size(); ++i)
    {
        res[i] = Gelu(input[i]);
    }
    
    return res;
}

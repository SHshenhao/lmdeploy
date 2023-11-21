// Copyright (c) 2023, DeepLink.
#include "diopirt_impl.h"


namespace dipu {

namespace diopirt_helper {

// template<typename T>
// void compare_htensor(T* base, T* created, int64_t length, float atol, float rtol) {
//     T maxabsatol = 0.0;
//     T minabsatol = 1.0;
//     T maxabsrtol = 0.0;
//     T minabsrtol = 1.0;
//     std::vector<T> atoldiff;
//     std::vector<T> rtoldiff;
//     float sum1 = 0.0;
//     float sum2 = 0.0;
//     for (int64_t i = 0; i < length; i++) {
//         sum1 += std::abs(*(base+i));
//         sum2 += std::abs(*(created+i));
//         T absatol = std::abs(*(base+i) - *(created+i));
//         T absrtol = std::abs(1.0 - std::abs(*(created+i) / *(base+i)));
//         if (absatol >= maxabsatol) maxabsatol = absatol;
//         if (absatol <= minabsatol) minabsatol = absatol;
//         if (absrtol >= maxabsrtol) maxabsrtol = absrtol;
//         if (absrtol <= minabsrtol) minabsrtol = absrtol;
//         if (absatol >= atol || rtol >= rtol) {
//             atoldiff.emplace_back(absatol);
//             rtoldiff.emplace_back(absrtol);
//         }
//     }
//     std::cout<<"diff num:"<<atoldiff.size()<<" maxabsatol:"<<maxabsatol<<" minabsatol:"<<minabsatol<<
//                 " maxabsrtol:"<<maxabsrtol<<" minabsrtol:"<<minabsrtol<<
//                 " sumabs1:"<<sum1<<" meanabs1:"<<sum1/length<<
//                 " sumabs2:"<<sum2<<" meanabs2:"<<sum2/length<< std::endl;
// }

// template void compare_htensor(float* base, float* created, int64_t length, float atol = 0.003, float rtol = 0.003);
// template void compare_htensor(half* base, half* created, int64_t length, float atol = 0.003, float rtol = 0.003);

}  // diopirt_helper

}  // dipu
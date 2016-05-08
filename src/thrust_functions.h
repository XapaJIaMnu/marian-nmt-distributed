#pragma once

#include <cmath>
#include <cublas_v2.h>   
#include <thrust/device_vector.h>
#include <thrust/functional.h>

namespace thrust
{
  namespace detail
  {
    namespace functional
    {
    
      // Ugly hacks, but it seems this is neccessary. 
      __host__ __device__
      float expf2(float x) {
        float clip = 16;
        if(x > clip)
          x = clip;
        if(x < -clip)
          x = -clip;
        return expf(x);
      }
      
      __host__ __device__
      float logf2(float x) {
        if(x < 10e-10)
          x = 10e-10;
        return logf(x);
      }
    
      template<typename T>
      struct unary_exp : public thrust::unary_function<T,T> {
        __host__ __device__
        T operator()(const T &x) const { return expf2(x); }
      };
      
      template<typename Eval>
      __host__ __device__
      actor<composite<unary_operator<unary_exp>, actor<Eval>>>
      Exp(const actor<Eval> &_1) {
        return compose(unary_operator<unary_exp>(), _1);
      }
      
      template<typename T>
      struct unary_log : public thrust::unary_function<T,T> {
        __host__ __device__
        T operator()(const T &x) const { return logf2(x); }
      };
      
      template<typename Eval>
      __host__ __device__
      actor<composite<unary_operator<unary_log>, actor<Eval>>>
      Log(const actor<Eval> &_1) {
        return compose(unary_operator<unary_log>(), _1);
      } 
      
      template<typename T>
      struct unary_sigma : public thrust::unary_function<T,T> {
        __host__ __device__
        T operator()(const T &x) const { return 1.0 / (1.0 + expf2(-x)); }
      };
      
      template<typename Eval>
      __host__ __device__
      actor<composite<unary_operator<unary_sigma>, actor<Eval>>>
      Sigma(const actor<Eval> &_1) {
        return compose(unary_operator<unary_sigma>(), _1);
      }
      
      template<typename T>
      struct unary_tanh : public thrust::unary_function<T,T> {
        __host__ __device__
        T operator()(const T &x) const { return tanhf(x); }
      };
      
      template<typename Eval>
      __host__ __device__
      actor<composite<unary_operator<unary_tanh>, actor<Eval>>>
      Tanh(const actor<Eval> &_1) {
        return compose(unary_operator<unary_tanh>(), _1);
      }
      
      template<typename T1, typename T2>
      __host__ __device__
      actor<composite<binary_operator<thrust::maximum>, actor<T1>, actor<T2>>>
      Max(const actor<T1> &_1, const actor<T2> &_2) {
        return compose(binary_operator<thrust::maximum>(),
                       make_actor(_1),
                       make_actor(_2));
      }
    }
  }
}
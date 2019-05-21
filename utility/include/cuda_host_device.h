#pragma once

// Use this macro to not get the compiler confused when parsing a header
// that can be used with CUDA but currently isn't.

#ifndef _HOST_DEVICE_
 #ifdef __CUDACC__
  #define _HOST_DEVICE_ __host__ __device__
 #else
  #define _HOST_DEVICE_
 #endif
#endif
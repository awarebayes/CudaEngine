//
// Created by dev on 7/14/22.
//

#ifndef COURSE_RENDERER_MATRIX_CUH
#define COURSE_RENDERER_MATRIX_CUH

template<int n, int m>
struct mat{
        float data[n*m] = {0, };
	    const int rows = n;
	    const int cols = m;
	    __host__ __device__ __forceinline__ float &at(int i, int j) { return data[i * n + j]; };
};

template<int n, int m, int l>
__device__ __host__ __forceinline__ mat<n, l> dot(const mat<n, m> &self, const mat<m, l> &other);

template<int n, int m>
__device__ __host__ __forceinline__ mat<m, n> transpose(const mat<n, m> &self);


__device__ __host__ __forceinline__ mat<3, 3> inverse(const mat<3, 3> &self);

#endif//COURSE_RENDERER_MATRIX_CUH

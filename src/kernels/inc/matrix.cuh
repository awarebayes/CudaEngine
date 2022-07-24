//
// Created by dev on 7/14/22.
//

#ifndef COURSE_RENDERER_MATRIX_CUH
#define COURSE_RENDERER_MATRIX_CUH
#include <cstdio>

template<int n, int m>
struct mat{
        float data[n*m] = {0, };
	    const int rows = n;
	    const int cols = m;
	    __host__ __device__ __forceinline__ float &at(int i, int j) { return data[i * m + j]; };
		[[nodiscard]] __host__ __device__ __forceinline__ float at(int i, int j) const { return data[i * m + j]; };
};


template<int n, int m, int l>
__device__ __host__ __forceinline__
mat<n, l> dot(const mat<n, m> &self, const mat<m, l> &other){
	mat<n, l> result{0, };
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < l; ++j) {
			for (int k = 0; k < m; ++k) {
				result.at(i, j) += self.at(i, k) * other.at(k, j);
			}
		}
	}
	return result;
}


template<int n, int m>
__device__ __host__ __forceinline__ mat<m, n> transpose(const mat<n, m> &self)
{
	mat<m, n> result{};
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < m; ++j)
			result.at(j, i) = self.at(i, j);
	return result;
}

__device__ __host__ __forceinline__ mat<3, 3> inverse(mat<3, 3> &self)
{
	mat<3, 3> result{};
	float determinant=0;
	for(int i=0;i<3;i++)
		determinant = determinant + (self.at(0, i)*(self.at(1, (i+1)%3)*self.at(2, (i+2)%3) - self.at(1, (i+2)%3)*self.at(2, (i+1)%3)));

	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			result.at(i, j) = ((self.at((i+1)%3, (j+1)%3) * self.at((i+2)%3, (j+2)%3)) -
			                   (self.at((i+1)%3, (j+2)%3)*self.at((i+2)%3, (j+1)%3)) / determinant);
	return result;
}

template<int n>
__device__ __host__ mat<n,n>  identity_matrix()
{
	mat<n, n> result{};
	for (int i = 0; i < n; ++i)
		result.at(i, i) = 1;
	return result;
}

__device__ __host__ __forceinline__ float3 m2v(const mat<4,1> &m)
{
	float w = m.at(3, 0);
	return {
			m.at(0, 0) / w,
			m.at(1, 0) / w,
			m.at(2, 0) / w
	};
}

__device__ __host__ __forceinline__ mat<4, 1> v2m(const float3 &v)
{
	return {v.x, v.y, v.z, 1.0f};
}

__device__ __host__ mat<4, 4> viewport(int x, int y, int w, int h, int depth);

__device__ __host__ void dbg_print(const mat<4, 4> &mat);
__device__ __host__ void dbg_print(const mat<4, 1> &mat);

#endif//COURSE_RENDERER_MATRIX_CUH

//
// Created by dev on 7/14/22.
//

#include "../inc/matrix.cuh"


template<int n, int m, int l>
__device__ __host__ __forceinline__ mat<n, l> dot(const mat<n, m> &self, const mat<m, l> &other){
	mat<n, l> result{};
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < l; ++j) {
			for (int k = 0; k < m; ++k) {
				result.at(i, j) += self.at(i, k) * other.at(k, j);
			}
		}
	}
}

template<int n, int m>
__device__ __host__ __forceinline__ mat<m, n> transpose(const mat<n, m> &self)
{
	mat<m, n> result{};
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < m; ++j)
			result.at(j, i) = self.at(i, j);
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
}


template<int n>
__device__ __host__ mat<n,n>  identity_matrix()
{
	mat<n, n> result{};
	for (int i = 0; i < n; ++i)
		result.at(i, i) = 1;
	return result;
}

__device__ __host__ float3 m2v(const mat<4,4> &m)
{
	float3 result{};
	float w = m.at(3, 0);
	result.x = m.at(0, 0) / w;
	result.y = m.at(1, 0) / w;
	result.z = m.at(2, 0) / w;
	return result;
}

__device__ __host__ mat<4, 4> v2m(const float3 &v)
{
	mat<4, 4> result = identity_matrix<4>();
	result.at(0, 0) = v.x;
	result.at(1, 0) = v.y;
	result.at(2, 0) = v.z;
	result.at(3, 0) = 1.0f;
	return result;
}

__device__ __host__ mat<4, 4> viewport(int x, int y, int w, int h, int depth)
{
	mat<4, 4> result = identity_matrix<4>();
	result.at(0, 0) = w / 2.0f;
	result.at(1, 1) = h / 2.0f;
	result.at(2, 2) = depth / 2.0f;
	result.at(0, 3) = x + w / 2.0f;
	result.at(1, 3) = y + h / 2.0f;
	result.at(2, 3) = depth / 2.0f;
	return result;
}

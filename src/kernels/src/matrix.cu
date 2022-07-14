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

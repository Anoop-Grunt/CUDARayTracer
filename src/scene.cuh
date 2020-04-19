#pragma once
#include "sphere.cuh"


class scene
{
public:
	__device__ scene(sphere** device_pointer, int count);
	__device__ ~scene();
	__device__ bool hit_full(ray r, sphere_hit_details& rec);
private:
	sphere** spheres;
	int s_count;
};


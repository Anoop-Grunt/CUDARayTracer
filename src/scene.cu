#include "scene.cuh"

__device__ scene::scene(sphere** device_pointer, int count)
	:spheres(device_pointer), s_count(count)
{
}

__device__ scene::~scene()
{
}

__device__ bool scene::hit_full(ray r, hit_record& rec)
{
	bool temp_hit = false;
	for (int i = 0; i < s_count; i++) {
		if (spheres[i]->hit(r, 0.0, FLT_MAX, rec)) {
			temp_hit = true;
			break;
		}
	}

	return  temp_hit;
}
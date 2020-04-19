#include "scene.cuh"

__device__ scene::scene(sphere** device_pointer, int count)
	:spheres(device_pointer), s_count(count)
{
}

__device__ scene::~scene()
{
}

__device__ bool scene::hit_full(ray r, sphere_hit_details& rec)
{
	sphere_hit_details temp_rec;
	float closest_so_far = FLT_MAX;
	bool temp_hit = false;
	for (int i = 0; i < s_count; i++) {
		if (spheres[i]->hit(r, 0.0, closest_so_far, temp_rec)) {
			temp_hit = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
			break;
		}
	}

	return  temp_hit;
}
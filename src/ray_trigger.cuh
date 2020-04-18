

#include "ray.cuh"




using namespace glm;

struct hit_record

{
	float t;
	vec3 p;
	vec3 normal;
};

class trigger_test
{
public:
	__device__ trigger_test();
	__device__ ~trigger_test();

	__device__ virtual bool hit(ray r, float t_min, float t_max, hit_record rec);

private:
};
#include "ray.cuh"
using namespace glm;
class camera {
public:

    __device__ camera();
    __device__ ray get_ray(float u, float v);
    __device__ ~camera();
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;

};

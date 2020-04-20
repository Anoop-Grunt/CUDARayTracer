#pragma once
#include "ray.cuh"
using namespace glm;
struct sphere_hit_details

{
    float t;
    vec3 p;
    vec3 normal;
    bool front_face;
    __device__ inline void orient_normal(ray r, const vec3 point_out) {
        front_face = dot(r.get_direction(), point_out) < 0;
        normal = front_face ? point_out : -point_out;
    }
    vec3 albedo;
  
};



class sphere
{
public:
    __device__ sphere(vec3 center, float r, vec3 albedo);
    __device__ ~sphere();
    __device__ bool hit(ray r, float tmin, float tmax, sphere_hit_details& rec);



private:

    vec3 center;
    float radius;
    vec3 albedo;

};






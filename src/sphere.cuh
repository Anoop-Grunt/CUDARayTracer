#pragma once
#include "ray.cuh"
using namespace glm;
struct hit_record

{
    float t;
    vec3 p;
    vec3 normal;
};



class sphere
{
public:
    __device__ sphere(vec3 center, float r);
    __device__ ~sphere();
    __device__ bool hit(ray r, float tmin, float tmax, hit_record& rec);



private:

    vec3 center;
    float radius;

};






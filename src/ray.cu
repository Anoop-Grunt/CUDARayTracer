#include "ray.cuh"

ray::ray(glm::vec3 origin, glm::vec3 direction)
	:origin(origin), direction(direction)
{

}

ray::~ray()
{
}

__device__ glm::vec3 ray::get_origin()
{
	return  origin;
}

__device__ glm::vec3 ray::get_direction()
{
	return  direction;
}

__device__ glm::vec3 ray::get_point_at_t(float t)
{
	glm::vec3 point = origin + t * direction;
	return  point;
}

#pragma once
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class ray
{
public:
	__device__ ray(glm::vec3 origin, glm::vec3 direction);
	__device__ ~ray();
	__device__ glm::vec3 get_origin();
	__device__ glm::vec3 get_direction();
	__device__ glm::vec3 get_point_at_t(float t);


private:
	glm::vec3 origin;
	glm::vec3 direction;
};


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<GL/glew.h>
#include<iostream>
#include <GLFW/glfw3.h>
#include "Shader.h"
#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "vert_array_quick_bind.h"
#include "Global_Bind_Test.h"
#include "freecam.h"
#include <functional>
#include <fstream>
#include <sstream>
#include "artefact.h"
#include <stb_image/stb_image.h>
#include "cudaGL.h"
#include "cuda_gl_interop.h"
#include <thrust/device_vector.h>
#include "Texture.h"
#include "ray.cuh"
#include "sphere.cuh"
#include "scene.cuh"
#include <float.h>
#include "ray_tracing_camera.cuh"

int sample_count = 100;

using namespace glm;
#define gpuCheckErrs(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
using namespace std;
freecam primary_cam;

void MouseControlWrapper(GLFWwindow* window, double mouse_x, double mouse_y) {
	primary_cam.mouse_handler(window, mouse_x, mouse_y);
}

void ScrollControlWrapper(GLFWwindow* window, double x_disp, double y_disp) {
	primary_cam.scroll_handler(window, x_disp, y_disp);
}

__global__ void make_scene(sphere** spheres, scene** dev_ptr, int count) {
	*dev_ptr = new scene(spheres, count);
}

__device__ vec3 pix_data3(ray r, unsigned char* sky, int su, int sv, scene** sc) {
	sphere_hit_details rec;
	bool hit = (*sc)->hit_full(r, rec);

	if (hit)
	{
		vec3 N = vec3(rec.normal.x, rec.normal.y, rec.normal.z);
		return 0.5f * vec3(N.x + 1, N.y + 1, N.z + 1);
	}

	else
	{
		vec3 sky_col;
		int index = sv * 1920 * 3 + su * 3;
		int r = (int)sky[index];
		float rc = (float)((float)r / 255);
		int g = (int)sky[index + 1];
		float gc = (float)((float)g / 255);
		int b = (int)sky[index + 2];
		float bc = (float)((float)b / 255);
		sky_col.x = rc;
		sky_col.y = gc;
		sky_col.z = bc;
		return sky_col;
	}
}

__global__ void render(unsigned char* pix_buff_loc, int max_x, int max_y, unsigned char* sky, scene** sc) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x * 4 + i * 4;
	auto u = float(i) / max_x;
	auto v = float(j) / max_y;
	camera c;
	ray r1 = c.get_ray(u, v);
	vec3 col = pix_data3(r1, sky, i, j, sc);
	unsigned char r = (int)(255 * col.x);
	unsigned char g = (int)(255 * col.y);
	unsigned char b = (int)(255 * col.z);
	pix_buff_loc[pixel_index + 0] = (int)r + 1;
	pix_buff_loc[pixel_index + 1] = (int)g;
	pix_buff_loc[pixel_index + 2] = (int)b;
	pix_buff_loc[pixel_index + 3] = 255;
}

__global__ void add_spheres(sphere** sph, int count) {
	*(sph) = new  sphere(vec3(-1.5f, 0.00005f, -4.5f), 0.5f);
	*(sph + 1) = new sphere(vec3(1.5f, 0.00005f, -4.5f), 0.5f);
	*(sph + 2) = new sphere(vec3(0.f, 0.5f, -4.5f), 0.5f);
}

int main()
{
	cudaSetDevice(0);

	GLFWwindow* window;
	if (!glfwInit())
		return -1;
	window = glfwCreateWindow(1920, 1080, "CUDA project", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glewInit();
	float vertices[] = {
		// positions          // colors           // texture coords
		1.f,  1.f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, // top right
		1.f, -1.f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, // bottom right
		-1.f, -1.f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
		-1.f,  1.f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left
	};
	unsigned int indices[] = {
		0, 1, 3,
		1, 2, 3
	};

	VertexBuffer vb(vertices, sizeof(vertices));
	IndexBuffer eb(indices, 6);
	VertexArray va;

	va.spec_vertex_size(8);
	va.add_layout_spec(3);
	va.add_layout_spec(3);
	va.add_layout_spec(2);
	va.AddBuffer(vb);
	Texture t;
	int width, height, nrChannels;
	width = 1920;
	height = 1080;
	nrChannels = 4;

	unsigned int pbo;
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * nrChannels * sizeof(GLubyte), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	cudaGraphicsResource* res;
	gpuCheckErrs(cudaGraphicsGLRegisterBuffer(&res, pbo, cudaGraphicsMapFlagsNone));
	gpuCheckErrs(cudaGraphicsMapResources(1, &res, 0));
	unsigned char* out_data;
	size_t num_bytes;
	gpuCheckErrs(cudaGraphicsResourceGetMappedPointer((void**)&out_data, &num_bytes, res));

	int tx = 8;//threads x
	int ty = 8;//threads y
	dim3 blocks(width / tx + 1, height / ty + 1);
	dim3 threads(tx, ty);

	//setting up the sky

	int w, h, n;
	stbi_set_flip_vertically_on_load(true);
	unsigned char* data = stbi_load("res/textures/sky3.jpg", &w, &h, &n, 0);
	unsigned char* sky;
	cudaMalloc(&sky, w * h * 3);
	cudaMemcpy(sky, data, w * h * 3, cudaMemcpyHostToDevice);

	//setting up the rest of the scene

	sphere** spheres;
	cudaMalloc(&spheres, sizeof(sphere*) * 3);
	add_spheres << < 1, 1 >> > (spheres, 3);

	scene** sc;
	cudaMalloc(&sc, sizeof(scene*));
	make_scene << < 1, 1 >> > (spheres, sc, 3);

	vec3 lower_left_corner(-1.6, -0.9, -1.0);
	vec3 horizontal(3.2, 0.0, 0.0);
	vec3 vertical(0.0, 1.8, 0.0);
	vec3 origin(0.0, 0.0, 0.0);
	render << <blocks, threads >> > (out_data, width, height, sky, sc);
	cudaGraphicsUnmapResources(1, &res);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
	t.use_pbo(width, height);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	Shader s("res/shaders/tex_basic.shader");
	glfwSetCursorPosCallback(window, MouseControlWrapper);
	glfwSetScrollCallback(window, ScrollControlWrapper);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDisable(GL_CULL_FACE);

	while (!glfwWindowShouldClose(window))
	{
		primary_cam.input_handler(window);
		glClearColor(0.f, 0.f, 0.f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		s.Bind();
		t.Bind();
		va.Bind();
		eb.Bind();
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		va.Unbind();
		s.Unbind();
		/* Swap front and back buffers */
		glfwSwapBuffers(window);

		/* Poll for and process events */
		glfwPollEvents();
	}
	glfwTerminate();
	return 0;
}
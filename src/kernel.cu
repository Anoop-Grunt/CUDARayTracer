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
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "freecam.h"
#include <functional>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <fstream>
#include <sstream>
#include "artefact.h"
#include <stb_image/stb_image.h>
#include "cudaGL.h"
#include "cuda_gl_interop.h"
#define TX 32
#define TY 32

__device__
unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

__global__
void distanceKernel(uchar4* d_out, int w, int h, int2 pos) {
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	const int r = blockIdx.y * blockDim.y + threadIdx.y;
	if ((c >= w) || (r >= h)) return; // Check if within image bounds
	const int i = c + r * w; // 1D indexing
	const int dist = sqrtf((c - pos.x) * (c - pos.x) + (r - pos.y) * (r - pos.y));
	const unsigned char intensity = clip(255 - dist);

	d_out[i].x = 255;
	d_out[i].y = 0;
	d_out[i].z = 255;
	d_out[i].w = 255;
}

void kernelLauncher(uchar4* d_out, int w, int h, int2 pos) {
	const dim3 blockSize(TX, TY);
	const dim3 gridSize = dim3((w + TX - 1) / TX, (h + TY - 1) / TY);
	distanceKernel << <gridSize, blockSize >> > (d_out, w, h, pos);
}

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
__global__ void square(int* devin, int* devout)
{
	int i = threadIdx.x;
	devout[i] = devin[i] * devin[i];
}

int main()
{
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
	unsigned int VBO, VAO, EBO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);
	unsigned int texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	int width, height, nrChannels;
	width = 100;
	height = 100;
	nrChannels = 4;

	vector<unsigned char> data2;
	for (int i = 0; i < height; i++) {
		for (int j = 0;j < width;j++) {
			data2.push_back(255);
			data2.push_back(0);
			data2.push_back(255);
			data2.push_back(255);
		}
	}

	unsigned int pbo;
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * nrChannels * sizeof(GLubyte), NULL, GL_DYNAMIC_DRAW);
	void* mappedBuffer = glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
	//memcpy(mappedBuffer, &data2[0], width * height * nrChannels * sizeof(GLubyte));
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);


	cudaGraphicsResource* res;
	gpuCheckErrs(cudaGraphicsGLRegisterBuffer(&res, pbo, cudaGraphicsMapFlagsNone));
	gpuCheckErrs(cudaGraphicsMapResources(1, &res, 0));
	uchar4* out_data;
	size_t num_bytes;
	gpuCheckErrs(cudaGraphicsResourceGetMappedPointer((void**)&out_data, &num_bytes, res));
	int2 loc = { width / 2, height / 2 };
	kernelLauncher(out_data, width, height, loc);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glGenerateMipmap(GL_TEXTURE_2D);

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
		glBindTexture(GL_TEXTURE_2D, texture);
		glBindVertexArray(VAO);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
		s.Unbind();

		/* Swap front and back buffers */
		glfwSwapBuffers(window);

		/* Poll for and process events */
		glfwPollEvents();
	}
	glfwTerminate();
	return 0;
}
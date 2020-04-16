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
#include "Texture.h"

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

__global__ void
paint(unsigned char* g_odata)
{
	int i = threadIdx.x;
	int off = i * 4;
	g_odata[off] = 180;  //red channel
	g_odata[off + 1] = 0;  //green channel
	g_odata[off + 2] = 255;   //blue channel
	g_odata[off + 3] = 255;   // alpha channel
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
	width = 5;
	height = 5;
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
	paint << <1, width* height >> > (out_data);
	cudaGraphicsUnmapResources(1, &res);
	unsigned char* h_in;
	h_in = (unsigned char*)malloc(width * height * nrChannels * sizeof(GLubyte));
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
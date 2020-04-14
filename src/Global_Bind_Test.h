#pragma once
#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "vert_array_quick_bind.h"
#include "GL/glew.h"

class mesh_prototype
{
public:
	mesh_prototype(float* vertices,unsigned int * indices, unsigned int vbsize, unsigned int ibsize);
	~mesh_prototype();
	void draw();
private:
	VertexArray va;
	unsigned int ibHandle;
	unsigned int ibsize;
	unsigned int* indices;
	IndexBuffer ib;
};





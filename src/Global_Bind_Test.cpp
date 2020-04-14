#include "Global_Bind_Test.h"
#include <iostream>

mesh_prototype::mesh_prototype(float* vertices,unsigned int* indices, unsigned int vbsize, unsigned int ibsize)
	: ibsize(ibsize), ibHandle(0), indices(indices), ib(IndexBuffer(indices, ibsize))
{
	VertexBuffer vb(vertices, vbsize);
	va.spec_vertex_size(6);
	va.add_layout_spec(3);
	va.add_layout_spec(3);
	va.AddBuffer(vb);
}

mesh_prototype::~mesh_prototype()
{

}

void mesh_prototype::draw()
{
	va.Bind();
	ib.Bind();
    glDrawElements(GL_TRIANGLES, ibsize, GL_UNSIGNED_INT, 0);
	va.Unbind();
	ib.Unbind();

}


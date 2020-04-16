#include "vert_array_quick_bind.h"
#include"GL/glew.h"
#include <iostream>

VertexArray::VertexArray()
	:last_index(0), offset(0), vertex_size(0)
{
	glGenVertexArrays(1, &m_rendererID);
	std::cout << "use spec_vertex_size() and add_layout_spec() before using AddBuffer()" << std::endl;
}

VertexArray::~VertexArray()
{
	//glDeleteVertexArrays(1, &m_rendererID);
}

void VertexArray::AddBuffer(const VertexBuffer vb)
{
	int vert_buffer_relapse;
	int  vert_array_relapse;
	glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &vert_buffer_relapse);
	glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vert_array_relapse);

	Bind();
	vb.Bind();
	for (int i = 0; i < layouts.size(); i++) {
		glEnableVertexAttribArray(last_index);
		glVertexAttribPointer(last_index, layouts[last_index], GL_FLOAT, GL_FALSE, vertex_size * sizeof(float), (void*)offset);
		std::cout << last_index << " " << layouts[last_index] << " " << vertex_size << " " << offset << std::endl;
		offset += layouts[last_index] * sizeof(float);
		last_index++;
	}
	Unbind();
	vb.Unbind();
	glBindBuffer(GL_ARRAY_BUFFER, vert_buffer_relapse);
	glBindVertexArray(vert_array_relapse);
}

void VertexArray::Bind() const
{
	glBindVertexArray(m_rendererID);
}

void VertexArray::Unbind() const

{
	glBindVertexArray(0);
}

void VertexArray::add_layout_spec(int floats_per_vert_attr)
{
	layouts.push_back(floats_per_vert_attr);
}

void VertexArray::spec_vertex_size(int floats_per_vert)
{
	vertex_size = floats_per_vert;
}
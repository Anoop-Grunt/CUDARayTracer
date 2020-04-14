#pragma once
#include "VertexBuffer.h"
#include <vector>
#include "IndexBuffer.h"

class VertexArray
{

public:

	VertexArray();
	~VertexArray();
	void AddBuffer(const VertexBuffer vb);
	void Bind() const;
	void Unbind() const;
	void add_layout_spec(int floats_per_vert_attr);
	void spec_vertex_size(int floats_per_vert);
	unsigned int m_rendererID;
	unsigned int last_index;
	unsigned int vertex_size;
	std::vector<int> layouts;
	int offset;

};


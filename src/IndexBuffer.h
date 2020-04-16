#pragma once

class IndexBuffer {
private:

	unsigned int m_count;
public:
	unsigned int m_RendererID;
	IndexBuffer(const unsigned int* data, unsigned int count);
	~IndexBuffer();

	void Bind()const;
	void Unbind()const;
	inline unsigned int getCount()const {
		return m_count;
	}
};
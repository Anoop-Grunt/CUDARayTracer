#pragma once
#include<sstream>
#include <string>

class Texture
{
public:
	Texture();
	~Texture();
	void Bind();
	void Unbind();
	void addImage(const char* path);
	void use_pbo(int width, int height);
private:
	unsigned int m_RendererID;
	int m_Height;
	int m_Width;
	unsigned char* tex_data;
	int m_NumChannels;
};


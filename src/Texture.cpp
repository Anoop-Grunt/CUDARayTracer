#include "Texture.h"
#include <GL/glew.h>
#include<stb_image/stb_image.h>
#include "Renderer.h"
#include <iostream>

Texture::Texture()
	:m_RendererID(0), m_Height(0), m_Width(0), tex_data(NULL)
{
	GLuint boundTexture = 0;
	glGetIntegerv(GL_TEXTURE_BINDING_2D, (GLint*)&boundTexture);
	glGenTextures(1, &m_RendererID);
	Bind();
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, boundTexture);
}

Texture::~Texture()
{
}

void Texture::Bind() 
{
	glBindTexture(GL_TEXTURE_2D, m_RendererID);
}

void Texture::Unbind()
{
	glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture::addImage(const char* path)
{
	GLuint boundTexture = 0;
	glGetIntegerv(GL_TEXTURE_BINDING_2D, (GLint*)&boundTexture);
	Bind();
	unsigned char* data = stbi_load(path, &m_Width, &m_Height, &m_NumChannels, 0);
	glCall(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_Width, m_Height, 0, GL_RGB, GL_UNSIGNED_BYTE, data));
	glGenerateMipmap(GL_TEXTURE_2D);
	Unbind();
	stbi_image_free(data);
	glBindTexture(GL_TEXTURE_2D, boundTexture);
}

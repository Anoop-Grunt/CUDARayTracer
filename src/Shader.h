#pragma once
#include <string>
#include "GL/glew.h"
#include "glm/mat4x4.hpp"

struct  ShaderProgramSource
{
	std::string vertex_shader_source;
	std::string fragment_shader_source;
};

class Shader
{
public:
	Shader(const std::string& filepath);
	~Shader();

	void Bind() const;
	void Unbind() const;

	//set uniforms

	void setUniform4f(const std::string& name, float f1, float f2, float f3, float f4);
	void setUniform4mat4f(const std::string& name, const glm::mat4 matrix);
	void setUniformfloat(const std::string& name, const float f1);
	
private:
	unsigned int m_RendererID;
	//uniforms cahcing
	unsigned int GetUniformLocation(const std::string& name);
	const std::string m_filepath;
	int CreateShader(std::string vertex_shader_source, std::string fragment_shader_source);
	unsigned int CompileShader(GLenum type, std::string source);
	ShaderProgramSource ParseShader(const std::string& filepath);
};



#include "Shader.h"
#include<GL/glew.h>
#include<iostream>
#include<sstream>
#include<fstream>
using namespace std;

Shader::Shader(const std::string& filepath)
	:m_filepath(filepath), m_RendererID(0)
{
	ShaderProgramSource source = ParseShader(filepath);
	string vertex_shader_src = source.vertex_shader_source;
	string fragment_shader_src = source.fragment_shader_source;
	m_RendererID = CreateShader(vertex_shader_src, fragment_shader_src);
}

Shader::~Shader()
{
	glDeleteProgram(m_RendererID);
}

void Shader::Bind() const
{
	glUseProgram(m_RendererID);
}

void Shader::Unbind() const
{
	glUseProgram(0);
}

void Shader::setUniform4f(const std::string& name, float f0, float f1, float f2, float f3)
{
	int location = GetUniformLocation(name);
	glUniform4f(location, f0, f1, f2, f3);
}

void Shader::setUniformfloat(const std::string& name, const float f1)
{
	int location = GetUniformLocation(name);
	glUniform1f(location, f1);
}

unsigned int Shader::GetUniformLocation(const std::string& name)
{
	int location = glGetUniformLocation(m_RendererID, name.c_str());
	if (location == -1) {
		std::cout << "warning: uniform " << name << " doesn't exist" << std::endl;
	}
	return location;
}

int Shader::CreateShader(std::string vertex_shader_source, std::string fragment_shader_source) {
	unsigned int program = glCreateProgram();
	unsigned int vertexShader = CompileShader(GL_VERTEX_SHADER, vertex_shader_source);
	unsigned int fragmentShader = CompileShader(GL_FRAGMENT_SHADER, fragment_shader_source);
	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);
	glLinkProgram(program);
	glValidateProgram(program);
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	return program;
}

unsigned int Shader::CompileShader(GLenum type, std::string source) {
	unsigned int shader = glCreateShader(type);
	const char* src = source.c_str();
	glShaderSource(shader, 1, &src, nullptr);
	glCompileShader(shader);
	// nullptr in the length parameter means tat the whole string is used as source code for the sahder
	//: ERROR CHECKING CODE:
	int result;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &result);
	if (result == GL_FALSE) {
		int length;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
		char* message = (char*)malloc(length * sizeof(char));
		glGetShaderInfoLog(shader, length, &length, message);
		std::cout << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") << " shader failed to compile" << std::endl;
		std::cout << message << std::endl;
		glDeleteShader(shader);
		return 0;
	}
	return shader;
}

ShaderProgramSource Shader::ParseShader(const std::string& filepath) {
	std::ifstream stream(filepath);

	enum class ShaderType
	{
		NONE = -1, VERTEX = 0, FRAGMENT = 1
	};
	ShaderType type = ShaderType::NONE;
	stringstream ss[2];
	string line;
	while (getline(stream, line)) {
		if (line.find("#shader") != string::npos) {
			if (line.find("vertex") != string::npos)
				type = ShaderType::VERTEX;
			else if (line.find("shader") != string::npos)
				type = ShaderType::FRAGMENT;
		}
		else
		{
			ss[int(type)] << line << "\n";
		}
	}
	return { ss[int(ShaderType::VERTEX)].str(), ss[int(ShaderType::FRAGMENT)].str() };
}

void Shader::setUniform4mat4f(const std::string& name, const glm::mat4 matrix)

{
	glUniformMatrix4fv(GetUniformLocation(name), 1, GL_FALSE, &matrix[0][0]);
}
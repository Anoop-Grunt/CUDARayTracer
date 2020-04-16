#include "Renderer.h"
#include <iostream>
using namespace std;

void GLClearError() {
	while (glGetError() != GL_NO_ERROR);
}

bool GLLogCall(const char* funct, const char* file, int line) {
	while (GLenum error = glGetError()) {
		cout << "OpenGL error (" << error << ")" << "function :" << funct << " file :" << file << " line : " << line << endl;
		return false;
	}
	return true;
}
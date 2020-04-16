#pragma once
#define ASSERT(x) if(!(x))__debugbreak();
#define glCall(x)GLClearError();x;ASSERT(GLLogCall(#x,__FILE__,__LINE__));
#include<GL/glew.h>
void GLClearError();
bool GLLogCall(const char* funct, const char* file, int line);


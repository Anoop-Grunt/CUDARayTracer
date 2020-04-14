#shader vertex
#version 460 core

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 v_color;
out vec4 vertex_color;
void main() {

    gl_Position = position;
    vertex_color = v_color;

};



#shader fragment
#version 460 core

out vec4 color;
in vec4 vertex_color;
void main() {

    color = vertex_color;

};
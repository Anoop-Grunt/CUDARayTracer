#shader vertex
#version 460 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 norm;
out vec3 normal;
out vec3 fragment_position;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out mat4 mod;
void main() {
	gl_Position = projection * view * model * vec4(position, 1.f);
	normal = norm;
	fragment_position = vec3(model * vec4(position, 1.f));
	mod = model;
};

#shader fragment
#version 460 core
uniform vec4 LightColor;
uniform vec4 light_position;
uniform vec4 Object_Color;
uniform float alpha;
uniform float ambient;
uniform vec4 Camera_Position;
uniform float spec_highlight_intensity;
out vec4 color;
in vec3 normal;
in mat4 mod;
in vec3 fragment_position;
void main() {
	vec3 Normal = mat3(transpose(inverse(mod))) * normal;//just in case we wanna rotate the object lol
	vec3 normal_norm = normalize(Normal);
	vec3 light_incidence = normalize(vec3(light_position) - fragment_position);
	vec3 ambient_factor = ambient * vec3(LightColor);
	float diffuse_component = max(dot(light_incidence, normal_norm), 0.0f);
	vec3 diffuse_factor = diffuse_component * vec3(LightColor);

	vec3 campos = vec3(Camera_Position);
	vec3 CamDirection = normalize(campos - fragment_position);
	vec3 reflec_dir = reflect(-light_incidence, normal_norm);
	float spec = pow(max(dot(CamDirection, reflec_dir), 0.0), 256);
	vec3 specular_factor = spec_highlight_intensity * spec * vec3(LightColor);

	vec3 final = (diffuse_factor + ambient_factor + specular_factor) * vec3(Object_Color);

	color = vec4(final, alpha);
};
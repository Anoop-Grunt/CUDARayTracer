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
out vec4 color;
in vec3 normal;
in vec3 fragment_position;
struct Material {
	vec4 ambient;
	vec4 diffuse;
	vec4 specular;
	float shininess;
};
struct Light {
	vec4 position;

	vec4 ambient;
	vec4 diffuse;
	vec4 specular;
};
uniform Material material;
uniform Light light;
uniform float alpha;
uniform float ambient;
uniform vec4 Camera_Position;
void main() {
	//ambient component first
	vec3 ambient = vec3(light.ambient) * vec3(material.ambient);
	//Then the diffused component
	vec3 Normal = normalize(normal);
	vec3 light_incidence = normalize(vec3(light.position) - vec3(fragment_position));
	//because when the dot is negative i.e the angle of incidence is greater than 90, the face isn't even exposed to the source
	float diff = max(dot(Normal, light_incidence), 0.0);
	vec3 diffuse = vec3(light.diffuse) * (diff * vec3(material.diffuse));
	//Finally the specular highlights
	vec3 view_direction = normalize(vec3(Camera_Position) - fragment_position);
	vec3 reflect_direction = reflect(-light_incidence, Normal);
	float spec_highlight = pow(max(dot(view_direction, reflect_direction), 0.0), material.shininess);
	vec3 specular = vec3(light.specular) * (spec_highlight *vec3( material.specular));
	vec3 frag_final_factor = ambient + diffuse + specular;

	color = vec4(frag_final_factor, alpha);
};
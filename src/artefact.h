#pragma once
#include<string.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include "Shader.h"
#include "Global_Bind_Test.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include  <map>


class scene_artefact
{
public:

	scene_artefact(std::string const& path);
	~scene_artefact();
	void illustrate(Shader shader, glm::mat4 model);
	//void setup(Shader s, int i);
	mesh_prototype cook(std::vector<float> v, std::vector<unsigned int> i);
	void quick_illustrate_unstable(Shader shader, glm::mat4 model);
private:
	std::vector<mesh_prototype> meshes;
	std::map<float, unsigned int>  blend_indices_approximate;
	std::vector<std::vector<float>> vertex_data;
	std::vector<std::vector<unsigned int>> index_data;
	std::vector<glm::vec4> diffs;
	std::vector<glm::vec4> ambs;
	std::vector<glm::vec4> specs;
	std::vector<float> alphas;
	std::vector<float> shines;
	int blend_control = 0;
	void solve_child(aiNode* node,const aiScene* scene);
};


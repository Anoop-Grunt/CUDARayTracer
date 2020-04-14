#include "artefact.h"
using namespace std;

scene_artefact::scene_artefact(std::string const& path)

	

{
	Assimp::Importer import;
	const aiScene* scene = import.ReadFile(path, aiProcess_Triangulate | aiProcess_FindInvalidData);

	for (int i = 0; i < scene->mRootNode->mNumChildren; i++) {
		aiNode* child = scene->mRootNode->mChildren[i];
		cout << "No of meshes in : " << child->mName.C_Str() << " : " << child->mNumMeshes << endl;
		cout << "No of vertices in : " << child->mName.C_Str() << " : " << scene->mMeshes[child->mMeshes[0]]->mNumVertices << endl;
		cout << "No of faces in : " << child->mName.C_Str() << " : " << scene->mMeshes[child->mMeshes[0]]->mNumFaces << endl;
		cout << "no of children in node : " << child->mNumChildren << endl;
		solve_child(child, scene);
	}
	for (int i = 0; i < alphas.size();i++) {
		blend_indices_approximate[alphas[i]] = i;
	}
	//make sure all the artefacts mesh-materials have different alphas

}

scene_artefact::~scene_artefact()
{
}

void scene_artefact::illustrate(Shader shader, glm::mat4 model)
{
	for (std::map<float, unsigned int>::reverse_iterator it = blend_indices_approximate.rbegin(); it != blend_indices_approximate.rend(); ++it) {
		int i = it->second;
		shader.setUniform4mat4f("model", model);
		shader.setUniformfloat("alpha", alphas[i]);
		shader.setUniform4f("material.ambient", specs[i].x, specs[i].y, specs[i].z, 1.f);
		shader.setUniform4f("material.diffuse", diffs[i].x, diffs[i].y, diffs[i].z, 1.f);
		shader.setUniform4f("material.specular", specs[i].x, specs[i].y, specs[i].z, 1.f);
		shader.setUniformfloat("material.shininess", 200.0);
		mesh_prototype m(&vertex_data[i][0], &index_data[i][0], vertex_data[i].size() * sizeof(float), index_data[i].size());
		m.draw();
	}
}

mesh_prototype scene_artefact::cook(std::vector<float> v, std::vector<unsigned int> i)
{
	mesh_prototype m(&v[0], &i[0], v.size() * sizeof(float), i.size());
	return m;
}

void scene_artefact::quick_illustrate_unstable(Shader shader, glm::mat4 model)
{
	for (std::map<float, unsigned int>::reverse_iterator it = blend_indices_approximate.rbegin(); it != blend_indices_approximate.rend(); ++it) {
		int i = it->second;
		shader.setUniform4mat4f("model", model);
		shader.setUniformfloat("alpha", alphas[i]);
		shader.setUniform4f("material.ambient", specs[i].x, specs[i].y, specs[i].z, 1.f);
		shader.setUniform4f("material.diffuse", diffs[i].x, diffs[i].y, diffs[i].z, 1.f);
		shader.setUniform4f("material.specular", specs[i].x, specs[i].y, specs[i].z, 1.f);
		shader.setUniformfloat("material.shininess", 200.0);
		meshes[i].draw();
		
	}
}

void scene_artefact::solve_child(aiNode* node,const aiScene* scene)
{
	blend_control += 1;
	for (int i = 0; i < node->mNumMeshes;i++) {
		aiMesh* mesh = scene->mMeshes[node->mMeshes[0]];
		vector<float> verts;
		vector<unsigned int> indices;

		for (int j = 0; j < mesh->mNumVertices;j++) {
			verts.push_back(mesh->mVertices[j].x);
			verts.push_back(mesh->mVertices[j].y);
			verts.push_back(mesh->mVertices[j].z);

			verts.push_back(mesh->mNormals[j].x);
			verts.push_back(mesh->mNormals[j].y);
			verts.push_back(mesh->mNormals[j].z);
		}

		for (int j = 0; j < mesh->mNumFaces; j++) {
			for (int k = 0; k < mesh->mFaces[j].mNumIndices; k++) {
				indices.push_back(mesh->mFaces[j].mIndices[k]);
			}
		}

		//mesh_prototype msh(&verts[0], &indices[0], verts.size() * sizeof(float), indices.size());

		aiColor3D color;
		aiMaterial* mat = scene->mMaterials[mesh->mMaterialIndex];
		mat->Get(AI_MATKEY_COLOR_DIFFUSE, color);
		glm::vec4 diff(color.r, color.g, color.b, 1.0);
		mat->Get(AI_MATKEY_COLOR_AMBIENT, color);
		glm::vec4 amb(color.r, color.g, color.b, 1.0);
		mat->Get(AI_MATKEY_COLOR_SPECULAR, color);
		glm::vec4 spec(color.r, color.g, color.b, 1.0);
		float shine;
		mat->Get(AI_MATKEY_SHININESS_STRENGTH, shine);
		float alpha;
		mat->Get(AI_MATKEY_OPACITY, alpha);
		alpha -= 0.000010f * blend_control;
		blend_control += 1;
		cout << endl;
		cout << diff.x << " " << diff.y << " " << diff.z << endl;
		cout << amb.x << " " << amb.y << " " << amb.z << endl;
		cout << spec.x << " " << spec.y << " " << spec.z << endl;
		cout << node->mName.C_Str() << endl;
		cout << endl;
		vertex_data.push_back(verts);
		index_data.push_back(indices);
		diffs.push_back(diff);
		specs.push_back(spec);
		ambs.push_back(amb);
		alphas.push_back(alpha);
		shines.push_back(shine);
		meshes.push_back(cook(verts, indices));
	}

	for (unsigned int i = 0; i < node->mNumChildren; i++)
	{
		solve_child(node->mChildren[i], scene);
	}
}
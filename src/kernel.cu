

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<GL/glew.h>
#include<iostream>
#include <GLFW/glfw3.h>
#include "Shader.h"
#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "vert_array_quick_bind.h"
#include "Global_Bind_Test.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "freecam.h"
#include <functional>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <fstream>
#include <sstream>
#include "artefact.h"
#include <stb_image/stb_image.h>


using namespace std;
freecam primary_cam;

void MouseControlWrapper(GLFWwindow* window, double mouse_x, double mouse_y) {
    primary_cam.mouse_handler(window, mouse_x, mouse_y);
}

void ScrollControlWrapper(GLFWwindow* window, double x_disp, double y_disp) {
    primary_cam.scroll_handler(window, x_disp, y_disp);
}







__global__ void square(int *devin, int * devout)
{
    int i = threadIdx.x;
    devout[i] = devin[i] * devin[i];
}

int main()
{
    int arr1[] = { 1,2,3 };
    
    
    int* dev_in;
    cudaError_t err =  cudaMalloc(&dev_in, sizeof(arr1));
    int* dev_out;
    err = cudaMalloc(&dev_out, sizeof(arr1));
    
    int* out;
    
    err =  cudaMemcpy(dev_in, arr1, sizeof(arr1), cudaMemcpyHostToDevice);
    /*std::cout << err << std::endl;*/

    square << < 1, 3 >> > (dev_in, dev_out);
    
    int* res;
    res = (int*)malloc(sizeof(arr1));
    err = cudaMemcpy(res, dev_out, sizeof(arr1), cudaMemcpyDeviceToHost);
   /* std::cout << err << std::endl; */
    for (int i = 0; i < 3; i++) {
        std::cout << res[i] << " ";
    }
    std::cout << std::endl << " yaaaay! it worked" << std::endl;

 



    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(1920, 1080, "CUDA project", glfwGetPrimaryMonitor(), NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }
    

    /* Make the window's0 context current */
    glfwMakeContextCurrent(window);
    glewInit();
    
    float vertices[] = {
        // positions          // colors           // texture coords
        1.f,  1.f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, // top right
        1.f, -1.f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, // bottom right
        -1.f, -1.f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
        -1.f,  1.f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left 
    };
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


    int width, height, nrChannels;
    // The FileSystem::getPath(...) is part of the GitHub repository so we can find files on any IDE/platform; replace it with your own image path.
    unsigned char* data = stbi_load("D:/CUDA/first_project/res/textures/RhoOphichius_60Da_70mm_50.jpg", &width, &height, &nrChannels, 0);
    if (data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);
    
    Shader s("res/shaders/tex_basic.shader");
    /*s.Bind();*/







    char path[] = "models/metroid/metroid.obj";

    char path2[] = "models/rocket/cool-rocket-painted.obj";

    char path3[] = "models/astroguy/astroguy.obj";

    char path4[] = "models/plane/plane.obj";

    char path5[] = "models/Text/text.obj";

    char path6[] = "models/flag/flag-painted.obj";



    Shader phong_shader("res/shaders/phong_prototype.shader");

    /*phong_shader.Bind();*/



    glm::mat4 model = glm::mat4(1.0f);

    model = glm::rotate(model, glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    model = glm::rotate(model, glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    model = glm::scale(model, glm::vec3(0.4f, 0.4f, 0.4f));

    scene_artefact s0(path3);

    scene_artefact s1(path);

    scene_artefact s2(path2);

    scene_artefact s3(path4);

    scene_artefact s4(path5);

    scene_artefact s5(path6);



    glm::mat4 model2 = glm::mat4(1.0f);

    model2 = glm::rotate(model2, glm::radians(120.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    model2 = glm::translate(model2, glm::vec3(-0.f, -3.0f, -0.f));

    model2 = glm::scale(model2, glm::vec3(1.4f, 1.4f, 1.4f));



    glm::mat4 model3 = glm::mat4(1.0f);

    model3 = glm::scale(model3, glm::vec3(0.25f, 0.25f, 0.25f));



    glm::mat4 model4 = glm::translate(model3, glm::vec3(16.0f, -0.f, -0.f));

    model4 = glm::rotate(model4, glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f));



    glm::mat4 model5 = glm::translate(model3, glm::vec3(0.0f, -0.f, -0.f));

    model5 = glm::translate(model5, glm::vec3(0.f, -12.0f, 0.f));



    glm::mat4 model6 = glm::mat4(1.0f);

    model6 = glm::translate(model, glm::vec3(-3.0f, 5.f, 0.f));

    model6 = glm::rotate(model6, glm::radians(90.0f), glm::vec3(1.f, 0.f, 0.f));

    model6 = glm::rotate(model6, glm::radians(180.0f), glm::vec3(0.f, 0.f, 1.f));



    glm::mat4 model7 = glm::mat4(1.f);

    model7 = glm::translate(model7, glm::vec3(6.f, -0.5f, -0.f));



    model4 = glm::translate(model4, glm::vec3(0.f, -2.f, 0.f));














    glfwSetCursorPosCallback(window, MouseControlWrapper);
    glfwSetScrollCallback(window, ScrollControlWrapper);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_CULL_FACE);


    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        primary_cam.input_handler(window);
       
        /* Render here */

        glClearColor(0.f, 0.f, 0.f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        s.Bind();
        glBindTexture(GL_TEXTURE_2D, texture);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        s.Unbind();
     




        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}



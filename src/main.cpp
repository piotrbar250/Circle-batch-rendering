#include <iostream>
#include <cstdlib>
#include <ctime>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include "Shader.h"
#include "BoidsRenderer.h"
using namespace std;

int main()
{
    srand(time(NULL));
    GLFWwindow* window;
    if(!glfwInit())
    {
        return -1;
    }

    float screenWidth = 1000;
    float screenHeight = 1000;

    window = glfwCreateWindow(screenWidth, screenHeight, "Window!", NULL, NULL);

    glfwMakeContextCurrent(window);

    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Cound't load opengl" << std::endl;
        glfwTerminate();
        return -1; 
    }

    Shader shader("../src/10.1.instancing.vs", "../src/10.1.instancing.fs");
    shader.use();
    shader.setVec3("u_Color", glm::vec3(1.0f, 0.0f, 0.0f));

    glm::mat4 projection = glm::ortho(0.0f, screenWidth, 0.0f, screenHeight); 
    glm::mat4 model = glm::mat4(1.0f);

    glm::mat4 mvp = projection * model;

    shader.setMat4("u_mvp", mvp);

    int boidsCount = 25;
    BoidsRenderer boidsRenderer(boidsCount);

    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        boidsRenderer.clear();

        boidsRenderer.update();

        boidsRenderer.draw();
        
        glfwSwapBuffers(window);
    }
    glfwTerminate();
    return 0;
}
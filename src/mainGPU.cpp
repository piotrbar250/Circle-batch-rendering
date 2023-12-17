#include <iostream>
#include <cstdlib>
#include <ctime>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include "global.hpp"
#include "Shader.h"
#include "CPU/BoidsRenderer.h"
#include "CPU/Boid.h"
#include "CPU/Flock.h"

#include "GPU/FlockGPU.h"
using namespace std;

int main()
{    
    // exit(1);
    GLFWwindow* window;
    if(!glfwInit())
    {
        return -1;
    }

    window = glfwCreateWindow(screenWidth, screenHeight, "Window!", NULL, NULL);

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

        if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Cound't load opengl" << std::endl;
        glfwTerminate();
        return -1; 
    }

    Shader shader("../src/10.1.instancing.vs", "../src/10.1.instancing.fs");
    shader.use();
    shader.setVec3("u_Color", glm::vec3(0.0f, 1.0f, 1.0f));

    glm::mat4 projection = glm::ortho(0.0f, screenWidth, 0.0f, screenHeight); 
    glm::mat4 model = glm::mat4(1.0f);

    glm::mat4 mvp = projection * model;

    shader.setMat4("u_mvp", mvp);

    // Computational part begins

    int boidsCount = 500;
    // Flock flock(boidsCount);
    // BoidsRenderer boidsRenderer(boidsCount, flock.translations);
    

    FlockGPU flockGPU(boidsCount);
    BoidsRenderer boidsRenderer(boidsCount, flockGPU.translations);

    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // flock.computeNextFrame();
        flockGPU.computeNextFrame();

        boidsRenderer.clear();

        boidsRenderer.update();

        boidsRenderer.draw();
        
        glfwSwapBuffers(window);
    }
    glfwTerminate();
    return 0;
}
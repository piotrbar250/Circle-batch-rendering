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
#include "GPU/FlockGridGPU.h"
#include "GPU/cuda_functions.h"
#include <chrono>

using namespace std;

void displayFPS();

int main()
{
    // exit(1);
    GLFWwindow *window;
    if (!glfwInit())
    {
        return -1;
    }

    window = glfwCreateWindow(screenWidth, screenHeight, "Window!", NULL, NULL);

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
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

    int boidsCount = 100;
    // Flock flock(boidsCount);
    // BoidsRenderer boidsRenderer(boidsCount, flock.translations);

    GridParams params;
    params.width = 1000;
    params.height = 1000;
    params.cellSize = 50; // TO BE CHANGED!!!!!
    params.widthCount = (params.width + params.cellSize - 1) / params.cellSize;
    params.heightCount = (params.height + params.cellSize - 1) / params.cellSize;
    params.cellCount = params.widthCount * params.heightCount;

    FlockGridGPU flockGridGPU(boidsCount, params);
    BoidsRenderer boidsRenderer(boidsCount, flockGridGPU.translations);

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // flock.computeNextFrame();
        flockGridGPU.computeNextFrame(&(boidsRenderer.cuda_vbo_resource), params);

        boidsRenderer.clear();

        // boidsRenderer.update();

        boidsRenderer.draw();

        glfwSwapBuffers(window);

        displayFPS();
    }
    glfwTerminate();
    return 0;
}

auto lastTime = std::chrono::high_resolution_clock::now();
int frameCount = 0;

void displayFPS()
{
    // Measure current time
    auto currentTime = std::chrono::high_resolution_clock::now();
    frameCount++;

    // Check if 5 seconds have passed
    if (std::chrono::duration_cast<std::chrono::seconds>(currentTime - lastTime).count() >= 2)
    {
        // Calculate and display FPS
        double fps = frameCount / 2.0;
        std::cout << "FPS: " << fps << std::endl;

        // Reset timer and frame count
        lastTime = currentTime;
        frameCount = 0;
    }
}
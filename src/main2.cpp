#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "Shader.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
using namespace std;

int main()
{
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

    glm::vec2 translations[25];
    int index = 0;
    for(int y = 0; y < 5; y++)
    {
        for(int x = 0; x < 5; x++)
        {
            glm::vec2 translation;
            translation.x = x * 200;
            translation.y = y * 200;
            translations[index++] = translation;     
        }
    }

    // for(int i = 0; i < 25; i++ )
    // {
    //     cout << i << " " << translations[i].x << " " << translations[i].y << endl;
    // }

    unsigned int instancevbo;
    glGenBuffers(1, &instancevbo);
    glBindBuffer(GL_ARRAY_BUFFER, instancevbo);
    glBufferData(GL_ARRAY_BUFFER, 25 * sizeof(glm::vec2), &translations[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);


    vector<float> quad = {
        100.0f, 100.0f,
        200.0f, 100.0f,
        200.0f, 200.0f,

        100.0f, 100.0f,
        200.0f, 200.0f,
        100.0f, 200.0f
    };

    unsigned int vao;
    unsigned int vbo;

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, quad.size() * sizeof(float), quad.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, instancevbo);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), (void*)0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glVertexAttribDivisor(1, 1);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    // glClearColor(0.25f, 0.5f, 0.75f, 1.0f);

    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        glClear(GL_COLOR_BUFFER_BIT);
        glBindVertexArray(vao);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, 25);
        glBindVertexArray(0);
        
        glfwSwapBuffers(window);
    }
    glfwTerminate();
    return 0;
}
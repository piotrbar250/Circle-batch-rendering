#include <iostream>
#include <cstdlib>
#include <ctime>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "Shader.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
using namespace std;

const float PI = 3.14159265359;

int randomIntFromNeg4To4() {
    // Initialize random seed
   

    // Generate random number between -4 and 4
    return rand() % 15 - 5;
}


int main()
{
float screenWidth = 1000;
float screenHeight = 1000;

    srand(time(NULL));
    GLFWwindow* window;
    if(!glfwInit())
    {
        return -1;
    }

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

    cout << screenHeight << endl;
    glm::mat4 projection = glm::ortho(0.0f, screenWidth, 0.0f, screenHeight); 
    glm::mat4 model = glm::mat4(1.0f);

    glm::mat4 mvp = projection * model;

    shader.setMat4("u_mvp", mvp);

    glm::vec2 translations[25];
    glm::vec2 directions[25];
    int index = 0;
    for(int y = 0; y < 5; y++)
    {
        for(int x = 0; x < 5; x++)
        {
            glm::vec2 translation;
            translation.x = randomIntFromNeg4To4();//x * 200;
            translation.y = randomIntFromNeg4To4();// * 200;
            translations[index] = translation;     
            directions[index++] = translation;
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


    // vector<float> quad = {
    //     100.0f, 100.0f,
    //     200.0f, 100.0f,
    //     200.0f, 200.0f,

    //     100.0f, 100.0f,
    //     200.0f, 200.0f,
    //     100.0f, 200.0f
    // };

    vector<float> vertices;

    float radius = 100;

    //this should in my opionion place the center of the circle in the middle of the screen 1000*1000 but it doesnt
    float ox = 500;
    float oy = 500;    
    
    // this apparently places it in the middle why, whats wrong??
    // float ox = 1000;
    // float oy = 1000;

    int num_of_segments = 4;
    float offset = 2*PI / float(num_of_segments);
    float angle = 0;

    for(int i = 0; i < num_of_segments; i++)
    {
        float cx = radius * cos(angle) + ox;
        float cy = radius * sin(angle) + oy;
        vertices.push_back(cx);
        vertices.push_back(cy);
        angle += offset;

        cout << "cx: " << cx << endl;
        cout << "cy: " << cy << endl;
        cout << endl << endl;
    }

    unsigned int vao;
    unsigned int vbo;

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);

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


        for(int i = 0; i < 25; i++)
        {
            translations[i].x += directions[i].x;
            translations[i].y += directions[i].y;
        }

        for(int i = 0; i < 25; i++)
        {
            translations[i].x = 0;
            translations[i].y = 0;
        }


        glBindBuffer(GL_ARRAY_BUFFER, instancevbo);
        glBufferSubData(GL_ARRAY_BUFFER,0, 25 * sizeof(glm::vec2), &translations[0]);
        glBindBuffer(GL_ARRAY_BUFFER, 0);


        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, num_of_segments, 1);
        glBindVertexArray(0);
        
        glfwSwapBuffers(window);
    }
    glfwTerminate();
    return 0;
}
#pragma once
#include <glad/glad.h>
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "../global.hpp"
#include "../GPU/cuda_functions.h"

using namespace std;
using namespace glm;

const float PI = 3.14159265359;

const glm::vec3 colorMap[] = {
    glm::vec3(0.0f, 0.5f, 1.0f),
    glm::vec3(0.2f, 1.0f, 0.2f),
    glm::vec3(0.9f, 0.1f, 0.1f),
    glm::vec3(0.7f, 0.3f, 1.0f),
    glm::vec3(1.0f, 0.84f, 0.0f),
};

class Circle
{
public:
    int numOfSegments;
    float radius;
    float cx;
    float cy;

    vector<float> vertices;

    Circle(int numOfSegments = NUMBER_OF_SEGMENTS, float radius = RADIUS, float cx = START_X, float cy = START_Y)
        : numOfSegments(numOfSegments), radius(radius), cx(cx), cy(cy)
    {
        float offset = 2 * PI / float(numOfSegments);
        float angle = 0;

        for (int i = 0; i < numOfSegments; i++)
        {
            float px = radius * cos(angle) + cx;
            float py = radius * sin(angle) + cy;
            vertices.push_back(px);
            vertices.push_back(py);
            angle += offset;
        }
    }
};

class BoidsRenderer
{
public:
    unsigned int VAO;
    unsigned int VBO;
    unsigned int instanceVBO;
    // unsigned int cuda_vbo_resource;
    void* cuda_vbo_resource; // cudaGraphicsResource*

    int boidsCount;
    vector<glm::vec2>& translations;
    vector<int>& colorIndex;
    vector<glm::vec3> boidColor;
    Circle circle;

    BoidsRenderer(int boidsCount, vector<glm::vec2>& translations, vector<int>& colorIndex) : boidsCount(boidsCount), translations(translations), colorIndex(colorIndex)
    {
        setColors();
        prepVertices();
    }

    ~BoidsRenderer()
    {
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &instanceVBO);
        glDeleteVertexArrays(1, &VAO);
    }

    void setColors()
    {
        boidColor.resize(boidsCount);
        for(int i = 0; i < boidsCount; i++)
        {
            boidColor[i] = colorMap[colorIndex[i]];
        }
    }

    void prepVertices()
    {
        circle = Circle();

        GLuint colorVBO;
        glGenBuffers(1, &colorVBO);
        glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
        glBufferData(GL_ARRAY_BUFFER, boidsCount * sizeof(glm::vec3), &boidColor[0], GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glGenBuffers(1, &instanceVBO);
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        // glBufferData(GL_ARRAY_BUFFER, boidsCount * sizeof(glm::vec2), &translations[0], GL_DYNAMIC_DRAW); DELETED
        glBufferData(GL_ARRAY_BUFFER, boidsCount * sizeof(glm::vec2), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, instanceVBO, cudaGraphicsMapFlagsWriteDiscard);
        cuda_functions::registerVBO(&cuda_vbo_resource, instanceVBO);

        glGenVertexArrays(1, &VAO);
        glBindVertexArray(VAO);

        glGenBuffers(1, &VBO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, circle.vertices.size() * sizeof(float), circle.vertices.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
        
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), (void*)0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glVertexAttribDivisor(1, 1);
        // glBindBuffer(GL_ARRAY_BUFFER, VBO); DELETED 

        glEnableVertexAttribArray(2);
        glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glVertexAttribDivisor(2, 1);

        glBindVertexArray(0);
    }

    void update()
    {
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferSubData(GL_ARRAY_BUFFER,0, boidsCount * sizeof(glm::vec2), &translations[0]);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void draw()
    {
        glBindVertexArray(VAO);
        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, NUMBER_OF_SEGMENTS, boidsCount);
        glBindVertexArray(0);
    }

    void clear()
    {
        glClear(GL_COLOR_BUFFER_BIT);
    }
};
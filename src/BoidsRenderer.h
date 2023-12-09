#include <glad/glad.h>
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "global.hpp"

using namespace std;
using namespace glm;

const float PI = 3.14159265359;

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

    int boidsCount;
    vector<glm::vec2>& translations;
    Circle circle;

    BoidsRenderer(int boidsCount, vector<glm::vec2>& translations) : boidsCount(boidsCount), translations(translations)
    {
        prepVertices();
    }

    ~BoidsRenderer()
    {
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &instanceVBO);
        glDeleteVertexArrays(1, &VAO);
    }

    void prepVertices()
    {
        circle = Circle();

        glGenBuffers(1, &instanceVBO);
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferData(GL_ARRAY_BUFFER, boidsCount * sizeof(glm::vec2), &translations[0], GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

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
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

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
#include <glad/glad.h>
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace std;
using namespace glm;

const float PI = 3.14159265359;

int randomInt() {
    return rand() % 15 - 5;
}

class Circle
{
public:
    int numOfSegments;
    float radius;
    float cx;
    float cy;

    vector<float> vertices;

    Circle(int numOfSegments = 50, float radius = 25, float cx = 100, float cy = 100)
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
    vector<glm::vec2> translations;
    Circle circle;

    BoidsRenderer(int boidsCount) : boidsCount(boidsCount)
    {
        prepVertices();
    }

    void prepVertices()
    {
        circle = Circle();
        translations.resize(boidsCount);

        int index = 0;
        for (int y = 0; y < 5; y++)
        {
            for (int x = 0; x < 5; x++)
            {
                glm::vec2 translation;
                translation.x = x * 200;
                translation.y = y * 200;
                translations[index++] = translation;
            }
        }

        glGenBuffers(1, &instanceVBO);
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferData(GL_ARRAY_BUFFER, boidsCount * sizeof(glm::vec2), &translations[0], GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glGenVertexArrays(1, &VAO);
        glBindVertexArray(VAO);

        glGenBuffers(1, &VBO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, circle.vertices.size() * sizeof(float), circle.vertices.data(), GL_DYNAMIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), (void*)0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glVertexAttribDivisor(1, 1);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
    }

    void update()
    {
        for(int i = 0; i < boidsCount; i++)
        {
            translations[i].x += randomInt();
            translations[i].y += randomInt();
        }        
        
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferSubData(GL_ARRAY_BUFFER,0, boidsCount * sizeof(glm::vec2), &translations[0]);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void draw()
    {
        glBindVertexArray(VAO);
        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 50, boidsCount);
        glBindVertexArray(0);
    }

    void clear()
    {
        glClear(GL_COLOR_BUFFER_BIT);
    }
};
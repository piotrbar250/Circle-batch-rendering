#include <glad/glad.h>
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace std;
using namespace glm;

const float PI = 3.14159265359;

class BoidsRenderer
{
public:
    int boidsCount;
    int numOfSegments = 50;
    vector<float> vertices; 

    BoidsRenderer(int boidsCount) : boidsCount(boidsCount)
    {
        
    }
};
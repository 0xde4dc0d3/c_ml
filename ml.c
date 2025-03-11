#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

typedef float train_sample[3];
train_sample TRAIN_DATA[] = {
    {0, 0, 0},
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 1},
};
#define TRAIN_SIZE (sizeof(TRAIN_DATA) / sizeof(TRAIN_DATA[0]))

float sigmoidf(float x) 
{
    return 1.f / (1.f + expf(-x));
}

float rand_float() 
{
    return (float)rand() / (float)RAND_MAX;
}

/* Takes the model as input (w1, w2, b) */
float model_cost(float w1, float w2, float b) 
{
    float cost = 0.f;

    for (size_t i = 0; i < TRAIN_SIZE; ++i) {
        float x1 = TRAIN_DATA[i][0];
        float x2 = TRAIN_DATA[i][1];
        float expected_y = TRAIN_DATA[i][2];
        float calculated_y = sigmoidf(w1 * x1 + w2 * x2 + b);
        float diff = expected_y - calculated_y;
        cost += diff * diff;
    }

    return cost /= TRAIN_SIZE;
}

int main(void)
{
    srand(69420);

    float bias  = rand_float();
    float w1    = rand_float();
    float w2    = rand_float();
    float cost = 0.f;

    printf("w1: %f, w2: %f, bias: %f\n", w1, w2, bias);

    for (size_t i = 0; i < TRAIN_SIZE; ++i) {
        float x1 = TRAIN_DATA[i][0];
        float x2 = TRAIN_DATA[i][1];
        float expected_y = TRAIN_DATA[i][2];
        float calculated_y = sigmoidf(w1 * x1 + w2 * x2 + bias);
        printf("cost: %f | %f & %f = %f | expected: %f\n", 
               model_cost(w1, w2, bias), x1, x2, calculated_y, expected_y);
    }

    printf("---------------------------------------------------------------\n");

    float learning_rate     = 1e-2;
    float eps               = 1e-3;
    int epochs              = 400*1000;

    for (size_t i = 0; i < epochs; ++i) {
        float cost = model_cost(w1, w1, bias);
    
        float dw1 = (model_cost(w1 + eps, w2, bias) - cost) / eps;
        float dw2 = (model_cost(w1, w2 + eps, bias) - cost) / eps;
        float db = (model_cost(w1, w2, bias + eps) - cost) / eps;
        
        w1   -= learning_rate * dw1;
        w2   -= learning_rate * dw2;
        bias -= learning_rate * db;
        //printf("%f\n", cost);
    }
    printf("w1: %f, w2: %f, bias: %f\n", w1, w2, bias);

    for (size_t i = 0; i < TRAIN_SIZE; ++i) {
        float x1 = TRAIN_DATA[i][0];
        float x2 = TRAIN_DATA[i][1];
        float expected_y = TRAIN_DATA[i][2];
        float calculated_y = sigmoidf(w1 * x1 + w2 * x2 + bias);
        printf("cost: %f | %f & %f = %f | expected: %f\n", 
               model_cost(w1, w2, bias), x1, x2, calculated_y, expected_y);
    }
    return 0;
}

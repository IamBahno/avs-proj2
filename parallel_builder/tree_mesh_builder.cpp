/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Ondřej Bahounek <xbahou00@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    2.12.2024
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

unsigned TreeMeshBuilder::processCube(Vec3_t<float> &cubeOffset,const ParametricScalarField &field, size_t gridSize)
{

    unsigned totalTriangles = 0;

    if(gridSize>1){
        Vec3_t<float> midPoint((cubeOffset.x + gridSize/2)*mGridResolution,
                                (cubeOffset.y + gridSize/2)*mGridResolution,
                                (cubeOffset.z + gridSize/2)*mGridResolution);

        float F = field.getIsoLevel() + (sqrt(3)/2)*(gridSize*mGridResolution);
        if ( evaluateFieldAt(midPoint, field) > F) {
            return 0;
        }

        for (int x =0; x < 2;x++){
            for (int y =0; y < 2;y++){
                for (int z =0; z < 2;z++){
                        #pragma omp task shared(totalTriangles)
                        {
                            Vec3_t<float> newOffset(cubeOffset.x + x * (gridSize/2), cubeOffset.y + y * (gridSize/2), cubeOffset.z + z * (gridSize/2));
                            unsigned triangles = processCube(newOffset,field, gridSize/2);

                            #pragma omp critical
                            totalTriangles += triangles;
                        }
                        
                }
            }
        }
    }
    else{
        unsigned tmp = buildCube(cubeOffset,field);
        #pragma omp critical
        totalTriangles += tmp;
    }


    #pragma omp taskwait
    return totalTriangles;
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.
    unsigned totalTriangles;
    Vec3_t<float> initialPosition(0,0,0);

    #pragma omp parallel
    #pragma omp single nowait
    {
        totalTriangles = processCube(initialPosition,field,mGridSize);
    }

    return totalTriangles;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
            // NOTE: This method is called from "buildCube(...)"!

    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
    for(unsigned i = 0; i < count; ++i)
    {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally take square root of the minimal square distance to get the real distance
    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    #pragma omp critical
    mTriangles.push_back(triangle);   
}

/**
 * @file    tree_mesh_builder.h
 *
 * @author  Ondřej Bahounek <xbahou00@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    2.12.2024
 **/

#ifndef TREE_MESH_BUILDER_H
#define TREE_MESH_BUILDER_H

#include "base_mesh_builder.h"

class TreeMeshBuilder : public BaseMeshBuilder
{
public:
    TreeMeshBuilder(unsigned gridEdgeSize);

protected:
    unsigned marchCubes(const ParametricScalarField &field);
    float evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field);
    void emitTriangle(const Triangle_t &triangle);
    const Triangle_t *getTrianglesArray() const { return mTriangles.data(); }

    unsigned processCube(Vec3_t<float> &offset,const ParametricScalarField &field, size_t gridSize);

    std::vector<Triangle_t> mTriangles;
};

#endif // TREE_MESH_BUILDER_H

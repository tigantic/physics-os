// Copyright 2025 Tigantic Labs. All Rights Reserved.

using System;
using UnityEngine;

namespace Tigantic.Ontic
{
    /// <summary>
    /// Field type enumeration matching Python FieldType.
    /// </summary>
    public enum FieldType
    {
        Scalar = 0,
        Vector = 1,
        Tensor = 2
    }

    /// <summary>
    /// Boundary condition type.
    /// </summary>
    public enum BoundaryCondition
    {
        Periodic = 0,
        Dirichlet = 1,
        Neumann = 2,
        Reflective = 3
    }

    /// <summary>
    /// Field statistics snapshot.
    /// </summary>
    [Serializable]
    public struct FieldStats
    {
        /// <summary>Maximum QTT rank.</summary>
        public int maxRank;

        /// <summary>Average QTT rank.</summary>
        public float avgRank;

        /// <summary>Number of QTT cores.</summary>
        public int numCores;

        /// <summary>Truncation error estimate.</summary>
        public float truncationError;

        /// <summary>Divergence norm (for vector fields).</summary>
        public float divergenceNorm;

        /// <summary>Field energy.</summary>
        public float energy;

        /// <summary>Compression ratio vs dense storage.</summary>
        public float compressionRatio;

        /// <summary>QTT memory usage in bytes.</summary>
        public long memoryBytes;

        /// <summary>Simulation step count.</summary>
        public int stepCount;

        /// <summary>State hash for provenance.</summary>
        public string stateHash;
    }

    /// <summary>
    /// Sample request for batch operations.
    /// </summary>
    [Serializable]
    public struct SampleRequest
    {
        /// <summary>Points to sample (normalized 0-1 coordinates).</summary>
        public Vector3[] points;

        /// <summary>Maximum rank for bounded sampling.</summary>
        public int maxRank;
    }

    /// <summary>
    /// Sample result from field.
    /// </summary>
    [Serializable]
    public struct SampleResult
    {
        /// <summary>Sampled values (scalar or vector components).</summary>
        public Vector4[] values;

        /// <summary>Actual rank used.</summary>
        public int rankUsed;

        /// <summary>Error estimate.</summary>
        public float errorEstimate;
    }

    /// <summary>
    /// Slice specification for 2D extraction.
    /// </summary>
    [Serializable]
    public struct SliceSpec
    {
        /// <summary>Slice axis (0=X, 1=Y, 2=Z).</summary>
        public int axis;

        /// <summary>Slice position (0-1 normalized).</summary>
        public float position;

        /// <summary>Output resolution.</summary>
        public Vector2Int resolution;

        /// <summary>Maximum rank for bounded slicing.</summary>
        public int maxRank;

        public SliceSpec(int axis, float position, Vector2Int resolution, int maxRank = 16)
        {
            this.axis = axis;
            this.position = position;
            this.resolution = resolution;
            this.maxRank = maxRank;
        }
    }

    /// <summary>
    /// Physics configuration for field simulation.
    /// </summary>
    [Serializable]
    public struct PhysicsConfig
    {
        /// <summary>Enable advection.</summary>
        public bool enableAdvection;

        /// <summary>Enable diffusion.</summary>
        public bool enableDiffusion;

        /// <summary>Enable pressure projection.</summary>
        public bool enableProjection;

        /// <summary>Viscosity coefficient.</summary>
        public float viscosity;

        /// <summary>Buoyancy strength.</summary>
        public float buoyancyStrength;

        /// <summary>Buoyancy direction.</summary>
        public Vector3 buoyancyDirection;

        /// <summary>External force field.</summary>
        public Vector3 externalForce;

        /// <summary>Vorticity confinement strength.</summary>
        public float vorticityConfinement;

        public static PhysicsConfig Default => new PhysicsConfig
        {
            enableAdvection = true,
            enableDiffusion = true,
            enableProjection = true,
            viscosity = 0.01f,
            buoyancyStrength = 0.0f,
            buoyancyDirection = Vector3.up,
            externalForce = Vector3.zero,
            vorticityConfinement = 0.0f
        };
    }

    /// <summary>
    /// Budget configuration for bounded mode.
    /// </summary>
    [Serializable]
    public struct BudgetConfig
    {
        /// <summary>Maximum rank budget.</summary>
        public int maxRank;

        /// <summary>Frame time budget in milliseconds.</summary>
        public float frameBudgetMs;

        /// <summary>Memory budget in MB.</summary>
        public float memoryBudgetMB;

        /// <summary>Error budget (max truncation error).</summary>
        public float errorBudget;

        /// <summary>Adaptive rank adjustment.</summary>
        public bool adaptiveRank;

        public static BudgetConfig Default => new BudgetConfig
        {
            maxRank = 32,
            frameBudgetMs = 8.0f,
            memoryBudgetMB = 64.0f,
            errorBudget = 0.01f,
            adaptiveRank = true
        };
    }
}

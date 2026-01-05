// Copyright 2025 Tigantic Labs. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "HyperTensorTypes.generated.h"

/**
 * Field type enumeration matching Python FieldType
 */
UENUM(BlueprintType)
enum class EHyperTensorFieldType : uint8
{
    Scalar UMETA(DisplayName = "Scalar Field"),
    Vector UMETA(DisplayName = "Vector Field"),
    Tensor UMETA(DisplayName = "Tensor Field"),
};

/**
 * Boundary condition type
 */
UENUM(BlueprintType)
enum class EHyperTensorBoundaryCondition : uint8
{
    Periodic UMETA(DisplayName = "Periodic"),
    Dirichlet UMETA(DisplayName = "Dirichlet (Fixed Value)"),
    Neumann UMETA(DisplayName = "Neumann (Fixed Gradient)"),
    Reflective UMETA(DisplayName = "Reflective"),
};

/**
 * Field statistics snapshot
 */
USTRUCT(BlueprintType)
struct HYPERTENSOR_API FHyperTensorFieldStats
{
    GENERATED_BODY()

    /** Maximum QTT rank */
    UPROPERTY(BlueprintReadOnly, Category = "HyperTensor")
    int32 MaxRank = 0;

    /** Average QTT rank */
    UPROPERTY(BlueprintReadOnly, Category = "HyperTensor")
    float AvgRank = 0.0f;

    /** Number of QTT cores */
    UPROPERTY(BlueprintReadOnly, Category = "HyperTensor")
    int32 NumCores = 0;

    /** Truncation error estimate */
    UPROPERTY(BlueprintReadOnly, Category = "HyperTensor")
    float TruncationError = 0.0f;

    /** Divergence norm (for vector fields) */
    UPROPERTY(BlueprintReadOnly, Category = "HyperTensor")
    float DivergenceNorm = 0.0f;

    /** Field energy */
    UPROPERTY(BlueprintReadOnly, Category = "HyperTensor")
    float Energy = 0.0f;

    /** Compression ratio vs dense storage */
    UPROPERTY(BlueprintReadOnly, Category = "HyperTensor")
    float CompressionRatio = 0.0f;

    /** QTT memory usage in bytes */
    UPROPERTY(BlueprintReadOnly, Category = "HyperTensor")
    int64 MemoryBytes = 0;

    /** Simulation step count */
    UPROPERTY(BlueprintReadOnly, Category = "HyperTensor")
    int32 StepCount = 0;

    /** State hash for provenance */
    UPROPERTY(BlueprintReadOnly, Category = "HyperTensor")
    FString StateHash;
};

/**
 * Field sample request for batch operations
 */
USTRUCT(BlueprintType)
struct HYPERTENSOR_API FHyperTensorSampleRequest
{
    GENERATED_BODY()

    /** Points to sample (normalized 0-1 coordinates) */
    UPROPERTY(BlueprintReadWrite, Category = "HyperTensor")
    TArray<FVector> Points;

    /** Maximum rank for bounded sampling */
    UPROPERTY(BlueprintReadWrite, Category = "HyperTensor")
    int32 MaxRank = 16;
};

/**
 * Field sample result
 */
USTRUCT(BlueprintType)
struct HYPERTENSOR_API FHyperTensorSampleResult
{
    GENERATED_BODY()

    /** Sampled values (scalar or vector components) */
    UPROPERTY(BlueprintReadOnly, Category = "HyperTensor")
    TArray<FVector4> Values;

    /** Actual rank used */
    UPROPERTY(BlueprintReadOnly, Category = "HyperTensor")
    int32 RankUsed = 0;

    /** Error estimate */
    UPROPERTY(BlueprintReadOnly, Category = "HyperTensor")
    float ErrorEstimate = 0.0f;
};

/**
 * Field slice specification for 2D extraction
 */
USTRUCT(BlueprintType)
struct HYPERTENSOR_API FHyperTensorSliceSpec
{
    GENERATED_BODY()

    /** Slice axis (0=X, 1=Y, 2=Z) */
    UPROPERTY(BlueprintReadWrite, Category = "HyperTensor")
    int32 Axis = 2;

    /** Slice position (0-1 normalized) */
    UPROPERTY(BlueprintReadWrite, Category = "HyperTensor")
    float Position = 0.5f;

    /** Output resolution */
    UPROPERTY(BlueprintReadWrite, Category = "HyperTensor")
    FIntPoint Resolution = FIntPoint(256, 256);

    /** Maximum rank for bounded slicing */
    UPROPERTY(BlueprintReadWrite, Category = "HyperTensor")
    int32 MaxRank = 16;
};

/**
 * Physics operator configuration
 */
USTRUCT(BlueprintType)
struct HYPERTENSOR_API FHyperTensorPhysicsConfig
{
    GENERATED_BODY()

    /** Enable advection */
    UPROPERTY(BlueprintReadWrite, Category = "HyperTensor|Physics")
    bool bEnableAdvection = true;

    /** Enable diffusion */
    UPROPERTY(BlueprintReadWrite, Category = "HyperTensor|Physics")
    bool bEnableDiffusion = true;

    /** Enable pressure projection */
    UPROPERTY(BlueprintReadWrite, Category = "HyperTensor|Physics")
    bool bEnableProjection = true;

    /** Viscosity coefficient */
    UPROPERTY(BlueprintReadWrite, Category = "HyperTensor|Physics")
    float Viscosity = 0.01f;

    /** Buoyancy strength */
    UPROPERTY(BlueprintReadWrite, Category = "HyperTensor|Physics")
    float BuoyancyStrength = 0.0f;

    /** Buoyancy direction */
    UPROPERTY(BlueprintReadWrite, Category = "HyperTensor|Physics")
    FVector BuoyancyDirection = FVector(0, 0, 1);

    /** External force field */
    UPROPERTY(BlueprintReadWrite, Category = "HyperTensor|Physics")
    FVector ExternalForce = FVector::ZeroVector;

    /** Vorticity confinement strength */
    UPROPERTY(BlueprintReadWrite, Category = "HyperTensor|Physics")
    float VorticityConfinement = 0.0f;
};

/**
 * Bounded mode budget configuration
 */
USTRUCT(BlueprintType)
struct HYPERTENSOR_API FHyperTensorBudgetConfig
{
    GENERATED_BODY()

    /** Maximum rank budget */
    UPROPERTY(BlueprintReadWrite, Category = "HyperTensor|Budget")
    int32 MaxRank = 32;

    /** Frame time budget in milliseconds */
    UPROPERTY(BlueprintReadWrite, Category = "HyperTensor|Budget")
    float FrameBudgetMs = 8.0f;

    /** Memory budget in MB */
    UPROPERTY(BlueprintReadWrite, Category = "HyperTensor|Budget")
    float MemoryBudgetMB = 64.0f;

    /** Error budget (max truncation error) */
    UPROPERTY(BlueprintReadWrite, Category = "HyperTensor|Budget")
    float ErrorBudget = 0.01f;

    /** Adaptive rank adjustment */
    UPROPERTY(BlueprintReadWrite, Category = "HyperTensor|Budget")
    bool bAdaptiveRank = true;
};

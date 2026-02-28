// Copyright 2025 Tigantic Labs. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Components/SceneComponent.h"
#include "OnticTypes.h"
#include "OnticFieldComponent.generated.h"

class UTextureRenderTarget2D;
class UVolumeTexture;

/**
 * Declaration for field update delegate
 */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnFieldUpdated, const FOnticFieldStats&, Stats);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnFieldSampled, const TArray<FVector>&, Points, const TArray<FVector4>&, Values);

/**
 * Ontic Field Component
 * 
 * Manages a QTT-compressed field within Unreal Engine.
 * Connects to Ontic Python backend for computation.
 */
UCLASS(ClassGroup=(Ontic), meta=(BlueprintSpawnableComponent))
class ONTIC_API UOnticFieldComponent : public USceneComponent
{
    GENERATED_BODY()

public:
    UOnticFieldComponent();

    // =========================================================================
    // LIFECYCLE
    // =========================================================================

    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType, 
                               FActorComponentTickFunction* ThisTickFunction) override;

    // =========================================================================
    // INITIALIZATION
    // =========================================================================

    /**
     * Initialize the field with given dimensions
     * @param SizeX Grid size in X dimension
     * @param SizeY Grid size in Y dimension  
     * @param SizeZ Grid size in Z dimension
     * @param FieldType Type of field (Scalar/Vector/Tensor)
     * @return True if initialization succeeded
     */
    UFUNCTION(BlueprintCallable, Category = "Ontic")
    bool Initialize(int32 SizeX, int32 SizeY, int32 SizeZ, 
                    EOnticFieldType FieldType = EOnticFieldType::Vector);

    /**
     * Initialize from a saved bundle file
     * @param BundlePath Path to .htf bundle file
     * @return True if load succeeded
     */
    UFUNCTION(BlueprintCallable, Category = "Ontic")
    bool InitializeFromBundle(const FString& BundlePath);

    /**
     * Check if field is initialized and ready
     */
    UFUNCTION(BlueprintPure, Category = "Ontic")
    bool IsInitialized() const { return bIsInitialized; }

    // =========================================================================
    // FIELD ORACLE API
    // =========================================================================

    /**
     * Sample field at given world positions
     * @param WorldPositions Array of world-space positions
     * @return Sample result with values and metadata
     */
    UFUNCTION(BlueprintCallable, Category = "Ontic|Sampling")
    FOnticSampleResult Sample(const TArray<FVector>& WorldPositions);

    /**
     * Sample field at a single world position
     * @param WorldPosition World-space position
     * @return Field value at position (Vector4 for all field types)
     */
    UFUNCTION(BlueprintPure, Category = "Ontic|Sampling")
    FVector4 SampleSingle(const FVector& WorldPosition) const;

    /**
     * Extract 2D slice from field
     * @param SliceSpec Slice specification
     * @param OutTexture Render target to write slice to
     * @return True if slice extraction succeeded
     */
    UFUNCTION(BlueprintCallable, Category = "Ontic|Sampling")
    bool Slice(const FOnticSliceSpec& SliceSpec, UTextureRenderTarget2D* OutTexture);

    /**
     * Get current field statistics
     * @return Field statistics snapshot
     */
    UFUNCTION(BlueprintPure, Category = "Ontic|Stats")
    FOnticFieldStats GetStats() const { return CachedStats; }

    // =========================================================================
    // SIMULATION
    // =========================================================================

    /**
     * Step the field simulation forward
     * @param DeltaTime Time step in seconds
     */
    UFUNCTION(BlueprintCallable, Category = "Ontic|Simulation")
    void Step(float DeltaTime);

    /**
     * Apply an impulse to the field
     * @param WorldPosition World-space position of impulse
     * @param Direction Direction of impulse (for vector fields)
     * @param Strength Impulse strength
     * @param Radius Impulse radius
     */
    UFUNCTION(BlueprintCallable, Category = "Ontic|Simulation")
    void ApplyImpulse(const FVector& WorldPosition, const FVector& Direction, 
                      float Strength, float Radius);

    /**
     * Apply a force field for one frame
     * @param ForceField Uniform force to apply
     */
    UFUNCTION(BlueprintCallable, Category = "Ontic|Simulation")
    void ApplyForce(const FVector& ForceField);

    /**
     * Set obstacle geometry from static mesh
     * @param Mesh Static mesh to use as obstacle
     * @param Transform World transform of obstacle
     */
    UFUNCTION(BlueprintCallable, Category = "Ontic|Simulation")
    void SetObstacle(UStaticMesh* Mesh, const FTransform& Transform);

    /**
     * Clear all obstacles
     */
    UFUNCTION(BlueprintCallable, Category = "Ontic|Simulation")
    void ClearObstacles();

    // =========================================================================
    // SERIALIZATION
    // =========================================================================

    /**
     * Serialize field to bundle file
     * @param BundlePath Output path for .htf bundle
     * @return True if serialization succeeded
     */
    UFUNCTION(BlueprintCallable, Category = "Ontic|Serialization")
    bool SaveToBundle(const FString& BundlePath);

    /**
     * Get field state hash for provenance
     * @return Deterministic hash of field state
     */
    UFUNCTION(BlueprintPure, Category = "Ontic|Serialization")
    FString GetStateHash() const { return CachedStats.StateHash; }

    // =========================================================================
    // VISUALIZATION
    // =========================================================================

    /**
     * Update a volume texture with field data
     * @param VolumeTexture Volume texture to update
     * @param MaxRank Maximum rank for bounded rendering
     * @return True if update succeeded
     */
    UFUNCTION(BlueprintCallable, Category = "Ontic|Visualization")
    bool UpdateVolumeTexture(UVolumeTexture* VolumeTexture, int32 MaxRank = 16);

    /**
     * Get color from field value using configured color map
     * @param Value Field value
     * @return Mapped color
     */
    UFUNCTION(BlueprintPure, Category = "Ontic|Visualization")
    FLinearColor ValueToColor(float Value) const;

    // =========================================================================
    // EVENTS
    // =========================================================================

    /** Called when field is updated (after Step) */
    UPROPERTY(BlueprintAssignable, Category = "Ontic|Events")
    FOnFieldUpdated OnFieldUpdated;

    /** Called when async sampling completes */
    UPROPERTY(BlueprintAssignable, Category = "Ontic|Events")
    FOnFieldSampled OnFieldSampled;

    // =========================================================================
    // CONFIGURATION
    // =========================================================================

    /** Field type */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Ontic|Config")
    EOnticFieldType FieldType = EOnticFieldType::Vector;

    /** Grid dimensions */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Ontic|Config")
    FIntVector GridSize = FIntVector(64, 64, 64);

    /** World-space bounds of the field */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Ontic|Config")
    FBox WorldBounds = FBox(FVector(-100, -100, -100), FVector(100, 100, 100));

    /** Boundary condition */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Ontic|Config")
    EOnticBoundaryCondition BoundaryCondition = EOnticBoundaryCondition::Periodic;

    /** Physics configuration */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Ontic|Physics")
    FOnticPhysicsConfig PhysicsConfig;

    /** Budget configuration for bounded mode */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Ontic|Budget")
    FOnticBudgetConfig BudgetConfig;

    /** Auto-step simulation each frame */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Ontic|Simulation")
    bool bAutoStep = false;

    /** Time scale for simulation */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Ontic|Simulation")
    float TimeScale = 1.0f;

protected:
    // =========================================================================
    // INTERNAL
    // =========================================================================

    /** Convert world position to normalized field coordinates */
    FVector WorldToFieldCoords(const FVector& WorldPos) const;

    /** Convert normalized field coordinates to world position */
    FVector FieldCoordsToWorld(const FVector& FieldCoords) const;

    /** Update cached statistics from Python backend */
    void UpdateCachedStats();

    /** Send command to Python backend */
    bool SendCommand(const FString& Command, const TArray<uint8>& Data);

    /** Receive response from Python backend */
    bool ReceiveResponse(TArray<uint8>& OutData);

private:
    /** Initialization flag */
    bool bIsInitialized = false;

    /** Cached field statistics */
    FOnticFieldStats CachedStats;

    /** Internal field handle (opaque pointer to native implementation) */
    void* NativeFieldHandle = nullptr;

    /** ZMQ socket for Python bridge (if enabled) */
    void* ZmqSocket = nullptr;

    /** Shared memory handle for high-performance bridge */
    void* SharedMemoryHandle = nullptr;
};

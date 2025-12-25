// Copyright 2025 Tigantic Labs. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "HyperTensorFieldComponent.h"
#include "HyperTensorFieldActor.generated.h"

class UProceduralMeshComponent;
class UMaterialInstanceDynamic;

/**
 * HyperTensor Field Actor
 * 
 * Convenience actor that wraps a HyperTensorFieldComponent with
 * optional visualization (bounding box, slices, isosurfaces).
 */
UCLASS(ClassGroup=(HyperTensor))
class HYPERTENSOR_API AHyperTensorFieldActor : public AActor
{
    GENERATED_BODY()

public:
    AHyperTensorFieldActor();

    virtual void BeginPlay() override;
    virtual void Tick(float DeltaTime) override;

    // =========================================================================
    // COMPONENTS
    // =========================================================================

    /** Field component (main logic) */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperTensor")
    UHyperTensorFieldComponent* FieldComponent;

    /** Scene root */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperTensor")
    USceneComponent* SceneRoot;

    /** Optional visualization mesh */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperTensor|Visualization")
    UProceduralMeshComponent* VisualizationMesh;

    // =========================================================================
    // VISUALIZATION SETTINGS
    // =========================================================================

    /** Show bounding box in editor */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HyperTensor|Visualization")
    bool bShowBoundingBox = true;

    /** Show field slices */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HyperTensor|Visualization")
    bool bShowSlices = false;

    /** Slice axis (0=X, 1=Y, 2=Z) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HyperTensor|Visualization", 
              meta = (EditCondition = "bShowSlices", ClampMin = 0, ClampMax = 2))
    int32 SliceAxis = 2;

    /** Slice position (0-1) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HyperTensor|Visualization",
              meta = (EditCondition = "bShowSlices", ClampMin = 0.0, ClampMax = 1.0))
    float SlicePosition = 0.5f;

    /** Visualization material */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HyperTensor|Visualization")
    UMaterialInterface* VisualizationMaterial;

    // =========================================================================
    // CONVENIENCE API
    // =========================================================================

    /**
     * Apply impulse at hit location
     * @param HitResult Hit result from trace
     * @param Strength Impulse strength
     * @param Radius Impulse radius
     */
    UFUNCTION(BlueprintCallable, Category = "HyperTensor")
    void ApplyImpulseAtHit(const FHitResult& HitResult, float Strength, float Radius);

    /**
     * Get field value at world position
     * @param WorldPosition World-space position
     * @return Field value (Vector4)
     */
    UFUNCTION(BlueprintPure, Category = "HyperTensor")
    FVector4 GetFieldValueAt(const FVector& WorldPosition) const;

protected:
    /** Update visualization mesh based on current settings */
    void UpdateVisualization();

    /** Create bounding box mesh */
    void CreateBoundingBoxMesh();

    /** Create slice plane mesh */
    void CreateSliceMesh();

private:
    /** Dynamic material instance for visualization */
    UPROPERTY()
    UMaterialInstanceDynamic* DynamicMaterial;
};

// Copyright 2025 Tigantic Labs. All Rights Reserved.

#include "HyperTensorFieldActor.h"
#include "ProceduralMeshComponent.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "DrawDebugHelpers.h"

AHyperTensorFieldActor::AHyperTensorFieldActor()
{
    PrimaryActorTick.bCanEverTick = true;

    // Create scene root
    SceneRoot = CreateDefaultSubobject<USceneComponent>(TEXT("SceneRoot"));
    RootComponent = SceneRoot;

    // Create field component
    FieldComponent = CreateDefaultSubobject<UHyperTensorFieldComponent>(TEXT("FieldComponent"));
    FieldComponent->SetupAttachment(RootComponent);

    // Create visualization mesh
    VisualizationMesh = CreateDefaultSubobject<UProceduralMeshComponent>(TEXT("VisualizationMesh"));
    VisualizationMesh->SetupAttachment(RootComponent);
    VisualizationMesh->SetCastShadow(false);
}

void AHyperTensorFieldActor::BeginPlay()
{
    Super::BeginPlay();

    // Setup dynamic material
    if (VisualizationMaterial)
    {
        DynamicMaterial = UMaterialInstanceDynamic::Create(VisualizationMaterial, this);
        VisualizationMesh->SetMaterial(0, DynamicMaterial);
    }

    UpdateVisualization();
}

void AHyperTensorFieldActor::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

#if WITH_EDITOR
    // Draw debug bounding box in editor
    if (bShowBoundingBox && FieldComponent->IsInitialized())
    {
        FBox Bounds = FieldComponent->WorldBounds;
        Bounds = Bounds.TransformBy(GetActorTransform());
        DrawDebugBox(GetWorld(), Bounds.GetCenter(), Bounds.GetExtent(),
                     FColor::Cyan, false, -1.0f, 0, 2.0f);
    }
#endif

    // Update slice visualization if enabled
    if (bShowSlices && DynamicMaterial)
    {
        DynamicMaterial->SetScalarParameterValue(TEXT("SlicePosition"), SlicePosition);
    }
}

void AHyperTensorFieldActor::ApplyImpulseAtHit(const FHitResult& HitResult, float Strength, float Radius)
{
    if (!FieldComponent->IsInitialized())
    {
        return;
    }

    // Calculate impulse direction from hit normal
    FVector Direction = -HitResult.ImpactNormal;
    
    FieldComponent->ApplyImpulse(HitResult.ImpactPoint, Direction, Strength, Radius);
}

FVector4 AHyperTensorFieldActor::GetFieldValueAt(const FVector& WorldPosition) const
{
    if (!FieldComponent->IsInitialized())
    {
        return FVector4(0, 0, 0, 0);
    }

    // Transform to actor local space
    FVector LocalPos = GetActorTransform().InverseTransformPosition(WorldPosition);
    
    return FieldComponent->SampleSingle(LocalPos);
}

void AHyperTensorFieldActor::UpdateVisualization()
{
    VisualizationMesh->ClearAllMeshSections();

    if (bShowBoundingBox)
    {
        CreateBoundingBoxMesh();
    }

    if (bShowSlices)
    {
        CreateSliceMesh();
    }
}

void AHyperTensorFieldActor::CreateBoundingBoxMesh()
{
    if (!FieldComponent->IsInitialized())
    {
        return;
    }

    FBox Bounds = FieldComponent->WorldBounds;
    FVector Min = Bounds.Min;
    FVector Max = Bounds.Max;

    // Create wireframe box vertices (8 corners)
    TArray<FVector> Vertices;
    Vertices.Add(FVector(Min.X, Min.Y, Min.Z)); // 0
    Vertices.Add(FVector(Max.X, Min.Y, Min.Z)); // 1
    Vertices.Add(FVector(Max.X, Max.Y, Min.Z)); // 2
    Vertices.Add(FVector(Min.X, Max.Y, Min.Z)); // 3
    Vertices.Add(FVector(Min.X, Min.Y, Max.Z)); // 4
    Vertices.Add(FVector(Max.X, Min.Y, Max.Z)); // 5
    Vertices.Add(FVector(Max.X, Max.Y, Max.Z)); // 6
    Vertices.Add(FVector(Min.X, Max.Y, Max.Z)); // 7

    // Create triangles for faces (wireframe would need line rendering)
    TArray<int32> Triangles;
    TArray<FVector> Normals;
    TArray<FVector2D> UVs;
    TArray<FColor> Colors;

    // For now, just create a simple visualization
    // Full implementation would use line rendering or wireframe material
}

void AHyperTensorFieldActor::CreateSliceMesh()
{
    if (!FieldComponent->IsInitialized())
    {
        return;
    }

    FBox Bounds = FieldComponent->WorldBounds;
    FVector Size = Bounds.GetSize();
    FVector Center = Bounds.GetCenter();

    // Create slice plane quad
    TArray<FVector> Vertices;
    TArray<int32> Triangles;
    TArray<FVector> Normals;
    TArray<FVector2D> UVs;

    float SliceCoord = FMath::Lerp(Bounds.Min[SliceAxis], Bounds.Max[SliceAxis], SlicePosition);

    // Generate quad vertices based on slice axis
    switch (SliceAxis)
    {
    case 0: // X-slice (YZ plane)
        Vertices.Add(FVector(SliceCoord, Bounds.Min.Y, Bounds.Min.Z));
        Vertices.Add(FVector(SliceCoord, Bounds.Max.Y, Bounds.Min.Z));
        Vertices.Add(FVector(SliceCoord, Bounds.Max.Y, Bounds.Max.Z));
        Vertices.Add(FVector(SliceCoord, Bounds.Min.Y, Bounds.Max.Z));
        Normals.Add(FVector(1, 0, 0));
        Normals.Add(FVector(1, 0, 0));
        Normals.Add(FVector(1, 0, 0));
        Normals.Add(FVector(1, 0, 0));
        break;

    case 1: // Y-slice (XZ plane)
        Vertices.Add(FVector(Bounds.Min.X, SliceCoord, Bounds.Min.Z));
        Vertices.Add(FVector(Bounds.Max.X, SliceCoord, Bounds.Min.Z));
        Vertices.Add(FVector(Bounds.Max.X, SliceCoord, Bounds.Max.Z));
        Vertices.Add(FVector(Bounds.Min.X, SliceCoord, Bounds.Max.Z));
        Normals.Add(FVector(0, 1, 0));
        Normals.Add(FVector(0, 1, 0));
        Normals.Add(FVector(0, 1, 0));
        Normals.Add(FVector(0, 1, 0));
        break;

    case 2: // Z-slice (XY plane)
    default:
        Vertices.Add(FVector(Bounds.Min.X, Bounds.Min.Y, SliceCoord));
        Vertices.Add(FVector(Bounds.Max.X, Bounds.Min.Y, SliceCoord));
        Vertices.Add(FVector(Bounds.Max.X, Bounds.Max.Y, SliceCoord));
        Vertices.Add(FVector(Bounds.Min.X, Bounds.Max.Y, SliceCoord));
        Normals.Add(FVector(0, 0, 1));
        Normals.Add(FVector(0, 0, 1));
        Normals.Add(FVector(0, 0, 1));
        Normals.Add(FVector(0, 0, 1));
        break;
    }

    // UVs
    UVs.Add(FVector2D(0, 0));
    UVs.Add(FVector2D(1, 0));
    UVs.Add(FVector2D(1, 1));
    UVs.Add(FVector2D(0, 1));

    // Triangles (two triangles for quad)
    Triangles.Add(0);
    Triangles.Add(1);
    Triangles.Add(2);
    Triangles.Add(0);
    Triangles.Add(2);
    Triangles.Add(3);

    // Create mesh section
    VisualizationMesh->CreateMeshSection(1, Vertices, Triangles, Normals, UVs,
                                          TArray<FColor>(), TArray<FProcMeshTangent>(), false);
}

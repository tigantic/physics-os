// Copyright 2025 Tigantic Labs. All Rights Reserved.

#include "HyperTensorFieldComponent.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/VolumeTexture.h"
#include "Engine/StaticMesh.h"

UHyperTensorFieldComponent::UHyperTensorFieldComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.bStartWithTickEnabled = false;
}

void UHyperTensorFieldComponent::BeginPlay()
{
    Super::BeginPlay();

    // Auto-initialize if grid size is set
    if (GridSize.X > 0 && GridSize.Y > 0 && GridSize.Z > 0)
    {
        Initialize(GridSize.X, GridSize.Y, GridSize.Z, FieldType);
    }
}

void UHyperTensorFieldComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    // Cleanup native resources
    if (NativeFieldHandle)
    {
        // Release native field
        NativeFieldHandle = nullptr;
    }

#if WITH_HYPERTENSOR_ZMQ
    if (ZmqSocket)
    {
        // Close ZMQ socket
        ZmqSocket = nullptr;
    }
#endif

#if WITH_HYPERTENSOR_SHMEM
    if (SharedMemoryHandle)
    {
        // Unmap shared memory
        SharedMemoryHandle = nullptr;
    }
#endif

    bIsInitialized = false;

    Super::EndPlay(EndPlayReason);
}

void UHyperTensorFieldComponent::TickComponent(float DeltaTime, ELevelTick TickType,
                                                FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    if (bIsInitialized && bAutoStep)
    {
        Step(DeltaTime * TimeScale);
    }
}

bool UHyperTensorFieldComponent::Initialize(int32 SizeX, int32 SizeY, int32 SizeZ,
                                             EHyperTensorFieldType InFieldType)
{
    if (bIsInitialized)
    {
        UE_LOG(LogTemp, Warning, TEXT("HyperTensorFieldComponent already initialized"));
        return false;
    }

    GridSize = FIntVector(SizeX, SizeY, SizeZ);
    FieldType = InFieldType;

    // Calculate bits per dimension (QTT representation)
    // bits = ceil(log2(size))
    int32 BitsX = FMath::CeilToInt(FMath::Log2(static_cast<float>(SizeX)));
    int32 BitsY = FMath::CeilToInt(FMath::Log2(static_cast<float>(SizeY)));
    int32 BitsZ = FMath::CeilToInt(FMath::Log2(static_cast<float>(SizeZ)));

    // TODO: Connect to Python backend via ZMQ or shared memory
    // For now, we initialize a local placeholder

#if WITH_HYPERTENSOR_ZMQ
    // Initialize ZMQ connection to Python backend
    // ZmqSocket = zmq_socket(zmq_ctx_new(), ZMQ_REQ);
    // zmq_connect(ZmqSocket, "tcp://localhost:5555");
#endif

    // Initialize cached stats
    CachedStats.MaxRank = BudgetConfig.MaxRank;
    CachedStats.NumCores = BitsX + BitsY + BitsZ;
    CachedStats.CompressionRatio = 100.0f; // Placeholder
    CachedStats.StepCount = 0;

    bIsInitialized = true;
    SetComponentTickEnabled(bAutoStep);

    UE_LOG(LogTemp, Log, TEXT("HyperTensorFieldComponent initialized: %dx%dx%d, Type=%d"),
           SizeX, SizeY, SizeZ, static_cast<int32>(FieldType));

    return true;
}

bool UHyperTensorFieldComponent::InitializeFromBundle(const FString& BundlePath)
{
    // TODO: Load bundle via Python backend
    UE_LOG(LogTemp, Warning, TEXT("InitializeFromBundle not yet implemented: %s"), *BundlePath);
    return false;
}

FHyperTensorSampleResult UHyperTensorFieldComponent::Sample(const TArray<FVector>& WorldPositions)
{
    FHyperTensorSampleResult Result;

    if (!bIsInitialized)
    {
        UE_LOG(LogTemp, Warning, TEXT("Cannot sample: field not initialized"));
        return Result;
    }

    Result.Values.Reserve(WorldPositions.Num());
    Result.RankUsed = BudgetConfig.MaxRank;
    Result.ErrorEstimate = 0.0f;

    for (const FVector& WorldPos : WorldPositions)
    {
        FVector4 Value = SampleSingle(WorldPos);
        Result.Values.Add(Value);
    }

    return Result;
}

FVector4 UHyperTensorFieldComponent::SampleSingle(const FVector& WorldPosition) const
{
    if (!bIsInitialized)
    {
        return FVector4(0, 0, 0, 0);
    }

    FVector FieldCoords = WorldToFieldCoords(WorldPosition);

    // Clamp to field bounds
    FieldCoords.X = FMath::Clamp(FieldCoords.X, 0.0f, 1.0f);
    FieldCoords.Y = FMath::Clamp(FieldCoords.Y, 0.0f, 1.0f);
    FieldCoords.Z = FMath::Clamp(FieldCoords.Z, 0.0f, 1.0f);

    // TODO: Actual QTT sampling via Python backend
    // For now, return a placeholder based on position
    float Dist = (FieldCoords - FVector(0.5f)).Size();
    float Value = FMath::Exp(-Dist * 4.0f);

    switch (FieldType)
    {
    case EHyperTensorFieldType::Scalar:
        return FVector4(Value, 0, 0, 0);
    case EHyperTensorFieldType::Vector:
        return FVector4(Value * (0.5f - FieldCoords.X),
                        Value * (0.5f - FieldCoords.Y),
                        Value * (0.5f - FieldCoords.Z),
                        0);
    case EHyperTensorFieldType::Tensor:
        return FVector4(Value, Value, Value, Value);
    default:
        return FVector4(0, 0, 0, 0);
    }
}

bool UHyperTensorFieldComponent::Slice(const FHyperTensorSliceSpec& SliceSpec,
                                        UTextureRenderTarget2D* OutTexture)
{
    if (!bIsInitialized || !OutTexture)
    {
        return false;
    }

    // TODO: Extract slice via Python backend and update render target
    UE_LOG(LogTemp, Warning, TEXT("Slice extraction not yet fully implemented"));
    return false;
}

void UHyperTensorFieldComponent::Step(float DeltaTime)
{
    if (!bIsInitialized)
    {
        return;
    }

    // TODO: Send step command to Python backend
    // For now, just increment step count
    CachedStats.StepCount++;

    // Broadcast update event
    OnFieldUpdated.Broadcast(CachedStats);
}

void UHyperTensorFieldComponent::ApplyImpulse(const FVector& WorldPosition,
                                               const FVector& Direction,
                                               float Strength, float Radius)
{
    if (!bIsInitialized)
    {
        return;
    }

    FVector FieldCoords = WorldToFieldCoords(WorldPosition);
    float NormalizedRadius = Radius / WorldBounds.GetSize().GetMax();

    // TODO: Send impulse command to Python backend
    UE_LOG(LogTemp, Log, TEXT("ApplyImpulse at (%f, %f, %f), Strength=%f, Radius=%f"),
           FieldCoords.X, FieldCoords.Y, FieldCoords.Z, Strength, NormalizedRadius);
}

void UHyperTensorFieldComponent::ApplyForce(const FVector& ForceField)
{
    if (!bIsInitialized)
    {
        return;
    }

    PhysicsConfig.ExternalForce = ForceField;
    // TODO: Send force update to Python backend
}

void UHyperTensorFieldComponent::SetObstacle(UStaticMesh* Mesh, const FTransform& Transform)
{
    if (!bIsInitialized || !Mesh)
    {
        return;
    }

    // TODO: Voxelize mesh and send obstacle mask to Python backend
    UE_LOG(LogTemp, Log, TEXT("SetObstacle: %s"), *Mesh->GetName());
}

void UHyperTensorFieldComponent::ClearObstacles()
{
    if (!bIsInitialized)
    {
        return;
    }

    // TODO: Clear obstacle mask in Python backend
    UE_LOG(LogTemp, Log, TEXT("ClearObstacles"));
}

bool UHyperTensorFieldComponent::SaveToBundle(const FString& BundlePath)
{
    if (!bIsInitialized)
    {
        return false;
    }

    // TODO: Request serialization from Python backend
    UE_LOG(LogTemp, Warning, TEXT("SaveToBundle not yet implemented: %s"), *BundlePath);
    return false;
}

bool UHyperTensorFieldComponent::UpdateVolumeTexture(UVolumeTexture* VolumeTexture, int32 MaxRank)
{
    if (!bIsInitialized || !VolumeTexture)
    {
        return false;
    }

    // TODO: Sample field into volume texture
    UE_LOG(LogTemp, Warning, TEXT("UpdateVolumeTexture not yet fully implemented"));
    return false;
}

FLinearColor UHyperTensorFieldComponent::ValueToColor(float Value) const
{
    // Default: blue-white-red diverging colormap
    float T = FMath::Clamp(Value, -1.0f, 1.0f) * 0.5f + 0.5f;
    
    if (T < 0.5f)
    {
        // Blue to white
        float S = T * 2.0f;
        return FLinearColor::LerpUsingHSV(FLinearColor(0.0f, 0.0f, 1.0f), 
                                           FLinearColor::White, S);
    }
    else
    {
        // White to red
        float S = (T - 0.5f) * 2.0f;
        return FLinearColor::LerpUsingHSV(FLinearColor::White,
                                           FLinearColor(1.0f, 0.0f, 0.0f), S);
    }
}

FVector UHyperTensorFieldComponent::WorldToFieldCoords(const FVector& WorldPos) const
{
    FVector Size = WorldBounds.GetSize();
    FVector Min = WorldBounds.Min;

    return FVector(
        (WorldPos.X - Min.X) / Size.X,
        (WorldPos.Y - Min.Y) / Size.Y,
        (WorldPos.Z - Min.Z) / Size.Z
    );
}

FVector UHyperTensorFieldComponent::FieldCoordsToWorld(const FVector& FieldCoords) const
{
    FVector Size = WorldBounds.GetSize();
    FVector Min = WorldBounds.Min;

    return FVector(
        Min.X + FieldCoords.X * Size.X,
        Min.Y + FieldCoords.Y * Size.Y,
        Min.Z + FieldCoords.Z * Size.Z
    );
}

void UHyperTensorFieldComponent::UpdateCachedStats()
{
    // TODO: Request stats from Python backend
}

bool UHyperTensorFieldComponent::SendCommand(const FString& Command, const TArray<uint8>& Data)
{
#if WITH_HYPERTENSOR_ZMQ
    // TODO: Implement ZMQ send
#endif
    return false;
}

bool UHyperTensorFieldComponent::ReceiveResponse(TArray<uint8>& OutData)
{
#if WITH_HYPERTENSOR_ZMQ
    // TODO: Implement ZMQ receive
#endif
    return false;
}

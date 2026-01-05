// Copyright 2025 Tigantic Labs. All Rights Reserved.

#include "HyperTensorModule.h"

#define LOCTEXT_NAMESPACE "FHyperTensorModule"

void FHyperTensorModule::StartupModule()
{
    // This code will execute after your module is loaded into memory.
    UE_LOG(LogTemp, Log, TEXT("HyperTensor Plugin loaded"));
}

void FHyperTensorModule::ShutdownModule()
{
    // This function may be called during shutdown to clean up your module.
    UE_LOG(LogTemp, Log, TEXT("HyperTensor Plugin unloaded"));
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FHyperTensorModule, HyperTensor)

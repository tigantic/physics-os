// Copyright 2025 Tigantic Labs. All Rights Reserved.

#include "OnticModule.h"

#define LOCTEXT_NAMESPACE "FOnticModule"

void FOnticModule::StartupModule()
{
    // This code will execute after your module is loaded into memory.
    UE_LOG(LogTemp, Log, TEXT("Ontic Plugin loaded"));
}

void FOnticModule::ShutdownModule()
{
    // This function may be called during shutdown to clean up your module.
    UE_LOG(LogTemp, Log, TEXT("Ontic Plugin unloaded"));
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FOnticModule, Ontic)

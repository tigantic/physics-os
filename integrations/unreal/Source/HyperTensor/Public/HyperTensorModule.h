// Copyright 2025 Tigantic Labs. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

class FHyperTensorModule : public IModuleInterface
{
public:
    /** IModuleInterface implementation */
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

    /**
     * Singleton-like access to this module's interface.
     *
     * @return Returns singleton instance, loading the module on demand if needed
     */
    static inline FHyperTensorModule& Get()
    {
        return FModuleManager::LoadModuleChecked<FHyperTensorModule>("HyperTensor");
    }

    /**
     * Checks to see if this module is loaded and ready.
     *
     * @return True if the module is loaded and ready to use
     */
    static inline bool IsAvailable()
    {
        return FModuleManager::Get().IsModuleLoaded("HyperTensor");
    }
};

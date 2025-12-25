// Copyright 2025 Tigantic Labs. All Rights Reserved.

using UnrealBuildTool;

public class HyperTensor : ModuleRules
{
    public HyperTensor(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicIncludePaths.AddRange(
            new string[] {
                // ... add public include paths required here ...
            }
        );

        PrivateIncludePaths.AddRange(
            new string[] {
                // ... add other private include paths required here ...
            }
        );

        PublicDependencyModuleNames.AddRange(
            new string[]
            {
                "Core",
                "CoreUObject",
                "Engine",
                "InputCore",
                "RenderCore",
                "RHI",
                "Niagara",
                "ProceduralMeshComponent",
            }
        );

        PrivateDependencyModuleNames.AddRange(
            new string[]
            {
                "Slate",
                "SlateCore",
                "Projects",
            }
        );

        // Enable ZMQ for Python bridge
        if (Target.Platform == UnrealTargetPlatform.Win64 ||
            Target.Platform == UnrealTargetPlatform.Linux)
        {
            PublicDefinitions.Add("WITH_HYPERTENSOR_ZMQ=1");
            // Add ZMQ library paths here
        }

        // Enable shared memory for high-performance local bridge
        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            PublicDefinitions.Add("WITH_HYPERTENSOR_SHMEM=1");
        }
    }
}

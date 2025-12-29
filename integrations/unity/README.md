# HyperTensor Unity Package

**Version**: 1.0.0  
**Unity Compatibility**: Unity 2021.3 LTS+  
**Package Type**: Runtime + Editor

## Overview

The HyperTensor Unity Package provides native C# integration for QTT field visualization and simulation within Unity projects.

## Features

- **FieldBehaviour**: MonoBehaviour for field management
- **Custom Inspectors**: Full editor integration with preview
- **VFX Graph Support**: Field-driven visual effects
- **Shader Graph**: Custom nodes for field sampling
- **Python Bridge**: Connect to HyperTensor Python backend

## Installation

### Option 1: Package Manager (Recommended)

1. Open Window → Package Manager
2. Click + → Add package from git URL
3. Enter: `https://github.com/tigantic/HyperTensor.git?path=/integrations/unity`

### Option 2: Manual

1. Copy the `com.tigantic.hypertensor` folder to your project's `Packages` directory
2. Unity will automatically detect and import the package

## Quick Start

### Basic Usage

```csharp
using Tigantic.HyperTensor;

public class FluidController : MonoBehaviour
{
    public HyperTensorField field;
    
    void Start()
    {
        // Initialize 64³ vector field
        field.Initialize(64, 64, 64, FieldType.Vector);
    }
    
    void Update()
    {
        // Step simulation
        field.Step(Time.deltaTime);
        
        // Sample field at position
        Vector4 value = field.Sample(transform.position);
    }
    
    void OnTriggerEnter(Collider other)
    {
        // Apply impulse
        field.ApplyImpulse(other.transform.position, other.velocity, 1.0f, 0.5f);
    }
}
```

### Editor Tools

1. Create a new GameObject
2. Add Component → HyperTensor → Field
3. Configure field properties in Inspector
4. Press Play to see simulation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Unity Engine                           │
├─────────────────────────────────────────────────────────────┤
│  C# API    │  Editor Tools    │  Shader Graph Nodes         │
├────────────┴─────────────────┴──────────────────────────────┤
│              HyperTensor Runtime Assembly                    │
├─────────────────────────────────────────────────────────────┤
│  FieldBehaviour  │  FieldRenderer  │  FieldVisualizer       │
├──────────────────┴────────────────┴─────────────────────────┤
│              Native Bridge (ZMQ/SharedMemory)                │
├─────────────────────────────────────────────────────────────┤
│              HyperTensor Python Backend                      │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

- Unity 2021.3 LTS or later
- .NET Standard 2.1
- Optional: Python 3.9+ with HyperTensor for backend computation

## License

Apache 2.0 - See LICENSE file

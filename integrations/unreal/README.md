# The Physics OS — Unreal Engine Plugin

**Version**: 1.0.0  
**Engine Compatibility**: Unreal Engine 5.3+  
**Plugin Type**: Runtime + Editor

## Overview

Ontic Unreal Plugin provides native Blueprint and C++ integration for QTT field visualization and simulation within Unreal Engine projects.

## Features

- **Field Actor**: Spawn and visualize Physics OS fields in 3D space
- **Blueprint Nodes**: Full Blueprint API for field manipulation
- **Material Integration**: Custom material functions for field-based shading
- **Niagara Support**: Field-driven particle systems
- **Python Bridge**: Connect to The Physics OS Python backend via ZMQ

## Installation

1. Copy the `The Physics OS` folder to your project's `Plugins` directory
2. Enable the plugin in Project Settings → Plugins
3. Restart the editor

## Quick Start

### Blueprint

1. Add a `OnticFieldActor` to your level
2. Configure field dimensions and resolution
3. Use Blueprint nodes to sample/evolve the field

### C++

```cpp
#include "The Ontic Engine/OnticField.h"

// Create field component
UOnticFieldComponent* Field = CreateDefaultSubobject<UOnticFieldComponent>(TEXT("Field"));
Field->Initialize(64, 64, 64, EOnticFieldType::Vector);

// Sample field
FVector Value = Field->Sample(FVector(0.5f, 0.5f, 0.5f));

// Step simulation
Field->Step(DeltaTime);
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Unreal Engine                             │
├─────────────────────────────────────────────────────────────┤
│  Blueprint API    │  C++ API    │  Editor Tools             │
├───────────────────┴─────────────┴───────────────────────────┤
│              Physics OS Plugin Runtime                      │
├─────────────────────────────────────────────────────────────┤
│  Field Actor  │  Field Component  │  Material Functions     │
├───────────────┴─────────────────┴───────────────────────────┤
│              Native Bridge (ZMQ/SharedMemory)                │
├─────────────────────────────────────────────────────────────┤
│              Physics OS Python Backend                      │
└─────────────────────────────────────────────────────────────┘
```

## License

Apache 2.0 - See LICENSE file

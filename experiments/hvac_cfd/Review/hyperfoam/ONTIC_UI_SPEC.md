# The Physics OS — UI Specification

**Document Version:** 1.0  
**Created:** January 18, 2026  
**Base Template:** creme-stack (Next.js 14 Enterprise Starter)  
**Target:** Production CFD Simulation Interface

---

## Executive Summary

This document specifies the complete UI implementation for The Physics OS/HyperFOAM, covering:

1. **Project Structure** — Adapting creme-stack to `ontic-ui/`
2. **Component Library** — Missing components for CFD workflows
3. **Dashboard Design** — Simulation control and visualization layout
4. **API Schema** — OpenAPI specification for HyperFOAM backend

---

## Part 1: Project Structure

### 1.1 Base Template Adaptation

**Source:** `HVAC_CFD/Review/creme-stack/`  
**Target:** `ontic-ui/`

```bash
# Initialize from creme-stack
cp -r HVAC_CFD/Review/creme-stack ontic-ui
cd ontic-ui

# Update package identity
npm pkg set name="ontic-ui"
npm pkg set version="0.1.0"
npm pkg set description="GPU-Accelerated CFD Simulation Interface"
```

### 1.2 Directory Structure

```
ontic-ui/
├── src/
│   ├── app/
│   │   ├── layout.tsx              # Root layout (Geist fonts, providers)
│   │   ├── page.tsx                # Landing → redirect to /dashboard
│   │   ├── globals.css             # Tailwind + CFD color tokens
│   │   ├── dashboard/
│   │   │   └── page.tsx            # Main CFD control panel
│   │   ├── simulations/
│   │   │   ├── page.tsx            # Simulation list
│   │   │   ├── [id]/
│   │   │   │   ├── page.tsx        # Simulation detail + viewer
│   │   │   │   ├── setup/page.tsx  # Mesh + BC configuration
│   │   │   │   └── results/page.tsx# Post-processing view
│   │   │   └── new/page.tsx        # Create simulation wizard
│   │   ├── meshes/
│   │   │   ├── page.tsx            # Mesh library
│   │   │   └── [id]/page.tsx       # Mesh detail + HyperGrid viewer
│   │   └── settings/
│   │       └── page.tsx            # User/compute preferences
│   │
│   ├── components/
│   │   ├── ui/                     # shadcn/ui base components
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── form.tsx
│   │   │   ├── input.tsx
│   │   │   ├── label.tsx
│   │   │   ├── table.tsx           # NEW: Data tables
│   │   │   ├── slider.tsx          # NEW: Parameter sliders
│   │   │   ├── tabs.tsx            # NEW: Tab navigation
│   │   │   ├── progress.tsx        # NEW: Simulation progress
│   │   │   ├── tooltip.tsx         # NEW: Tooltips
│   │   │   ├── select.tsx          # NEW: Dropdowns
│   │   │   ├── dialog.tsx          # NEW: Modal dialogs
│   │   │   ├── sheet.tsx           # NEW: Side panels
│   │   │   └── skeleton.tsx        # NEW: Loading states
│   │   │
│   │   ├── cfd/                    # CFD-specific components
│   │   │   ├── MeshViewer.tsx      # Three.js HyperGrid viewer
│   │   │   ├── VelocityField.tsx   # Vector field visualization
│   │   │   ├── TemperatureMap.tsx  # Scalar field heatmap
│   │   │   ├── ResidualChart.tsx   # Convergence plot (Recharts)
│   │   │   ├── BoundaryEditor.tsx  # BC configuration panel
│   │   │   ├── SimulationCard.tsx  # Simulation status card
│   │   │   ├── GeometryTree.tsx    # CSG operation tree
│   │   │   └── ColorLegend.tsx     # Field value legend
│   │   │
│   │   ├── layout/                 # Layout components
│   │   │   ├── Sidebar.tsx         # Navigation sidebar
│   │   │   ├── Header.tsx          # Top bar with user menu
│   │   │   ├── DashboardShell.tsx  # Main dashboard wrapper
│   │   │   └── ViewerPanel.tsx     # Resizable 3D viewer panel
│   │   │
│   │   └── simulation/             # Simulation control components
│   │       ├── ParameterForm.tsx   # Solver parameters
│   │       ├── RunControls.tsx     # Start/Stop/Pause buttons
│   │       ├── ProgressBar.tsx     # Iteration progress
│   │       └── LogViewer.tsx       # Real-time solver output
│   │
│   ├── hooks/
│   │   ├── index.ts                # Base hooks (from creme-stack)
│   │   ├── useSimulation.ts        # Simulation state & WebSocket
│   │   ├── useHyperGrid.ts         # Mesh loading & manipulation
│   │   ├── useFieldData.ts         # Field tensor streaming
│   │   └── useResiduals.ts         # Convergence data hook
│   │
│   ├── lib/
│   │   ├── api/
│   │   │   ├── client.ts           # openapi-fetch client
│   │   │   └── schema.d.ts         # Generated from OpenAPI
│   │   ├── providers/
│   │   │   ├── index.tsx           # Root providers
│   │   │   ├── query-provider.tsx  # TanStack Query
│   │   │   ├── theme-provider.tsx  # Light/dark mode
│   │   │   └── socket-provider.tsx # NEW: WebSocket context
│   │   ├── three/                  # NEW: Three.js utilities
│   │   │   ├── HyperGridGeometry.ts# Mesh → Three.js geometry
│   │   │   ├── VectorArrows.ts     # Velocity arrows
│   │   │   ├── IsoSurface.ts       # Scalar isosurfaces
│   │   │   └── ColorMap.ts         # Scientific colormaps
│   │   └── utils.ts                # cn() and helpers
│   │
│   ├── stores/
│   │   ├── index.ts                # Zustand stores
│   │   ├── simulationStore.ts      # Active simulation state
│   │   ├── viewerStore.ts          # 3D viewer camera/settings
│   │   └── uiStore.ts              # Sidebar, modals, theme
│   │
│   └── types/
│       ├── index.ts                # Common types
│       ├── simulation.ts           # Simulation interfaces
│       ├── mesh.ts                 # HyperGrid types
│       └── fields.ts               # CFD field types
│
├── public/
│   ├── colormaps/                  # Viridis, Jet, Coolwarm PNGs
│   └── icons/                      # CFD-specific icons
│
├── e2e/                            # Playwright tests
├── .storybook/                     # Component documentation
├── openapi.yaml                    # API specification
└── package.json
```

### 1.3 Package.json Updates

```json
{
  "name": "ontic-ui",
  "version": "0.1.0",
  "description": "GPU-Accelerated CFD Simulation Interface",
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "test": "vitest",
    "test:e2e": "playwright test",
    "storybook": "storybook dev -p 6006",
    "generate-api": "openapi-typescript openapi.yaml -o src/lib/api/schema.d.ts"
  },
  "dependencies": {
    "next": "14.2.x",
    "@tanstack/react-query": "^5.x",
    "zustand": "^5.x",
    "react-hook-form": "^7.x",
    "@hookform/resolvers": "^3.x",
    "zod": "^3.x",
    "openapi-fetch": "^0.9.x",
    
    "@radix-ui/react-dialog": "^1.x",
    "@radix-ui/react-dropdown-menu": "^2.x",
    "@radix-ui/react-select": "^2.x",
    "@radix-ui/react-slider": "^1.x",
    "@radix-ui/react-tabs": "^1.x",
    "@radix-ui/react-tooltip": "^1.x",
    "@radix-ui/react-progress": "^1.x",
    
    "@react-three/fiber": "^8.x",
    "@react-three/drei": "^9.x",
    "three": "^0.160.x",
    
    "recharts": "^2.x",
    "lucide-react": "^0.300.x",
    "sonner": "^1.x",
    "class-variance-authority": "^0.7.x",
    "clsx": "^2.x",
    "tailwind-merge": "^2.x"
  },
  "devDependencies": {
    "@types/three": "^0.160.x",
    "openapi-typescript": "^6.x",
    "vitest": "^1.x",
    "@playwright/test": "^1.x",
    "storybook": "^8.x"
  }
}
```

---

## Part 2: Missing Components

### 2.1 shadcn/ui Additions

Run these commands to add required components:

```bash
# Form controls
npx shadcn-ui@latest add table
npx shadcn-ui@latest add slider
npx shadcn-ui@latest add tabs
npx shadcn-ui@latest add select
npx shadcn-ui@latest add checkbox
npx shadcn-ui@latest add radio-group

# Feedback
npx shadcn-ui@latest add progress
npx shadcn-ui@latest add tooltip
npx shadcn-ui@latest add alert

# Layout
npx shadcn-ui@latest add dialog
npx shadcn-ui@latest add sheet
npx shadcn-ui@latest add separator
npx shadcn-ui@latest add scroll-area
npx shadcn-ui@latest add resizable

# Data display
npx shadcn-ui@latest add skeleton
npx shadcn-ui@latest add badge
npx shadcn-ui@latest add avatar
```

### 2.2 CFD-Specific Components

#### 2.2.1 MeshViewer.tsx

```tsx
'use client';

import { Canvas } from '@react-three/fiber';
import { OrbitControls, GizmoHelper, GizmoViewport } from '@react-three/drei';
import { useHyperGrid } from '@/hooks/useHyperGrid';
import { HyperGridMesh } from './HyperGridMesh';
import { BoundaryPatches } from './BoundaryPatches';

interface MeshViewerProps {
  meshId: string;
  showSDF?: boolean;
  showPatches?: boolean;
  colorField?: 'vol_frac' | 'sdf' | 'temperature' | 'velocity_mag';
}

export function MeshViewer({ 
  meshId, 
  showSDF = false,
  showPatches = true,
  colorField = 'vol_frac'
}: MeshViewerProps) {
  const { data: mesh, isLoading } = useHyperGrid(meshId);

  if (isLoading) return <MeshViewerSkeleton />;

  return (
    <div className="h-full w-full rounded-lg border bg-black">
      <Canvas camera={{ position: [2, 2, 2], fov: 50 }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1} />
        
        <HyperGridMesh 
          mesh={mesh} 
          colorField={colorField}
          opacity={0.8}
        />
        
        {showPatches && <BoundaryPatches patches={mesh.patches} />}
        
        <OrbitControls makeDefault />
        <GizmoHelper alignment="bottom-right" margin={[80, 80]}>
          <GizmoViewport />
        </GizmoHelper>
        
        <gridHelper args={[10, 10, '#444', '#222']} />
      </Canvas>
    </div>
  );
}
```

#### 2.2.2 ResidualChart.tsx

```tsx
'use client';

import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { useResiduals } from '@/hooks/useResiduals';

interface ResidualChartProps {
  simulationId: string;
}

export function ResidualChart({ simulationId }: ResidualChartProps) {
  const { data: residuals } = useResiduals(simulationId);

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer>
        <LineChart data={residuals} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <XAxis 
            dataKey="iteration" 
            label={{ value: 'Iteration', position: 'bottom' }}
          />
          <YAxis 
            scale="log" 
            domain={['auto', 'auto']}
            label={{ value: 'Residual', angle: -90, position: 'left' }}
          />
          <Tooltip />
          <Line 
            type="monotone" 
            dataKey="continuity" 
            stroke="#ef4444" 
            name="Continuity"
            dot={false}
          />
          <Line 
            type="monotone" 
            dataKey="momentum_x" 
            stroke="#22c55e" 
            name="Ux"
            dot={false}
          />
          <Line 
            type="monotone" 
            dataKey="momentum_y" 
            stroke="#3b82f6" 
            name="Uy"
            dot={false}
          />
          <Line 
            type="monotone" 
            dataKey="energy" 
            stroke="#f59e0b" 
            name="Energy"
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

#### 2.2.3 BoundaryEditor.tsx

```tsx
'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';

const boundarySchema = z.object({
  name: z.string().min(1),
  face: z.enum(['x-', 'x+', 'y-', 'y+', 'z-', 'z+']),
  type: z.enum(['inlet', 'outlet', 'wall', 'symmetry']),
  velocity: z.tuple([z.number(), z.number(), z.number()]).optional(),
  temperature: z.number().optional(),
});

type BoundaryForm = z.infer<typeof boundarySchema>;

interface BoundaryEditorProps {
  onSubmit: (data: BoundaryForm) => void;
  initialData?: Partial<BoundaryForm>;
}

export function BoundaryEditor({ onSubmit, initialData }: BoundaryEditorProps) {
  const form = useForm<BoundaryForm>({
    resolver: zodResolver(boundarySchema),
    defaultValues: initialData,
  });

  const watchType = form.watch('type');

  return (
    <Card>
      <CardHeader>
        <CardTitle>Boundary Condition</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
          <Input {...form.register('name')} placeholder="Patch name" />
          
          <Select onValueChange={(v) => form.setValue('face', v as any)}>
            <SelectTrigger>
              <SelectValue placeholder="Select face" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="x-">X- (Left)</SelectItem>
              <SelectItem value="x+">X+ (Right)</SelectItem>
              <SelectItem value="y-">Y- (Front)</SelectItem>
              <SelectItem value="y+">Y+ (Back)</SelectItem>
              <SelectItem value="z-">Z- (Bottom)</SelectItem>
              <SelectItem value="z+">Z+ (Top)</SelectItem>
            </SelectContent>
          </Select>

          <Select onValueChange={(v) => form.setValue('type', v as any)}>
            <SelectTrigger>
              <SelectValue placeholder="BC type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="inlet">Inlet (Dirichlet)</SelectItem>
              <SelectItem value="outlet">Outlet (Neumann)</SelectItem>
              <SelectItem value="wall">Wall (No-slip)</SelectItem>
              <SelectItem value="symmetry">Symmetry</SelectItem>
            </SelectContent>
          </Select>

          {watchType === 'inlet' && (
            <div className="grid grid-cols-3 gap-2">
              <Input type="number" step="0.01" placeholder="Ux" 
                {...form.register('velocity.0', { valueAsNumber: true })} />
              <Input type="number" step="0.01" placeholder="Uy"
                {...form.register('velocity.1', { valueAsNumber: true })} />
              <Input type="number" step="0.01" placeholder="Uz"
                {...form.register('velocity.2', { valueAsNumber: true })} />
            </div>
          )}

          {(watchType === 'inlet' || watchType === 'wall') && (
            <Input type="number" step="0.1" placeholder="Temperature (K)"
              {...form.register('temperature', { valueAsNumber: true })} />
          )}

          <Button type="submit" className="w-full">Add Boundary</Button>
        </form>
      </CardContent>
    </Card>
  );
}
```

### 2.3 WebSocket Provider

```tsx
// src/lib/providers/socket-provider.tsx
'use client';

import { createContext, useContext, useEffect, useState, ReactNode } from 'react';

interface SocketContextValue {
  socket: WebSocket | null;
  connected: boolean;
  subscribe: (channel: string, callback: (data: any) => void) => () => void;
}

const SocketContext = createContext<SocketContextValue | null>(null);

export function SocketProvider({ children }: { children: ReactNode }) {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [listeners] = useState(new Map<string, Set<(data: any) => void>>());

  useEffect(() => {
    const ws = new WebSocket(process.env.NEXT_PUBLIC_WS_URL ?? 'ws://localhost:8001');
    
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (event) => {
      const { channel, data } = JSON.parse(event.data);
      listeners.get(channel)?.forEach((cb) => cb(data));
    };

    setSocket(ws);
    return () => ws.close();
  }, []);

  const subscribe = (channel: string, callback: (data: any) => void) => {
    if (!listeners.has(channel)) listeners.set(channel, new Set());
    listeners.get(channel)!.add(callback);
    
    // Send subscription message
    socket?.send(JSON.stringify({ action: 'subscribe', channel }));
    
    return () => {
      listeners.get(channel)?.delete(callback);
      socket?.send(JSON.stringify({ action: 'unsubscribe', channel }));
    };
  };

  return (
    <SocketContext.Provider value={{ socket, connected, subscribe }}>
      {children}
    </SocketContext.Provider>
  );
}

export function useSocket() {
  const ctx = useContext(SocketContext);
  if (!ctx) throw new Error('useSocket must be used within SocketProvider');
  return ctx;
}
```

---

## Part 3: CFD Dashboard Design

### 3.1 Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│ ┌──────┐  The Ontic Engine                    [🔍 Search] [👤 User] [⚙️]    │
│ │ LOGO │  GPU-Accelerated CFD                                          │
│ └──────┘                                                                │
├─────────┬───────────────────────────────────────────────────────────────┤
│         │                                                               │
│ 📊 Dash │  ┌─────────────────────────────────────────────────────────┐ │
│         │  │                    ACTIVE SIMULATION                    │ │
│ 🔬 Sims │  │  ┌───────────────────┬─────────────────────────────┐   │ │
│         │  │  │                   │ Status: ● Running            │   │ │
│ 🧊 Mesh │  │  │                   │ Iteration: 2,340 / 10,000    │   │ │
│         │  │  │   3D VIEWER       │ Time: 0.234s                 │   │ │
│ 📈 Post │  │  │   (MeshViewer)    │ Δt: 1e-4                     │   │ │
│         │  │  │                   │ ─────────────────────────────│   │ │
│ ⚙️ Set  │  │  │   [vol_frac]      │ Performance:                 │   │ │
│         │  │  │   [sdf]           │ • 47.8 Mcells/s              │   │ │
│         │  │  │   [temperature]   │ • GPU: 78%                   │   │ │
│         │  │  │   [velocity]      │ • VRAM: 8.2 / 24 GB          │   │ │
│         │  │  │                   │                              │   │ │
│         │  │  └───────────────────┴─────────────────────────────┘   │ │
│         │  │                                                         │ │
│         │  │  ┌─────────────────────────────────────────────────┐   │ │
│         │  │  │              CONVERGENCE HISTORY                │   │ │
│         │  │  │  (ResidualChart)                                │   │ │
│         │  │  │                                                 │   │ │
│         │  │  │   [Continuity]  [Ux]  [Uy]  [Uz]  [Energy]     │   │ │
│         │  │  └─────────────────────────────────────────────────┘   │ │
│         │  │                                                         │ │
│         │  │  ┌──────────────┬──────────────┬──────────────────┐   │ │
│         │  │  │ ▶️ Start     │ ⏸️ Pause     │ ⏹️ Stop          │   │ │
│         │  │  └──────────────┴──────────────┴──────────────────┘   │ │
│         │  └─────────────────────────────────────────────────────────┘ │
│         │                                                               │
└─────────┴───────────────────────────────────────────────────────────────┘
```

### 3.2 Simulation Setup Page

```
┌─────────────────────────────────────────────────────────────────────────┐
│ ← Back to Simulations          New Simulation Setup                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─ Step 1: Geometry ─────────────────────────────────────────────────┐ │
│  │                                                                     │ │
│  │  [Upload STL/OBJ]     [Select from Library]     [Create HyperGrid] │ │
│  │                                                                     │ │
│  │  ┌─────────────────────────────────────────────────────────────┐   │ │
│  │  │                                                             │   │ │
│  │  │                      3D PREVIEW                             │   │ │
│  │  │                      (MeshViewer)                           │   │ │
│  │  │                                                             │   │ │
│  │  └─────────────────────────────────────────────────────────────┘   │ │
│  │                                                                     │ │
│  │  Grid Resolution:  Nx [128] × Ny [64] × Nz [64]                    │ │
│  │  Domain Size:      Lx [10.0] × Ly [5.0] × Lz [5.0] meters          │ │
│  │                                                                     │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─ Step 2: Boundary Conditions ──────────────────────────────────────┐ │
│  │                                                                     │ │
│  │  ┌───────────────────────────┐  ┌─────────────────────────────┐   │ │
│  │  │ BOUNDARY LIST             │  │ BOUNDARY EDITOR             │   │ │
│  │  │                           │  │ (BoundaryEditor component)  │   │ │
│  │  │ ● inlet_left   [Inlet]   │  │                             │   │ │
│  │  │ ● outlet_right [Outlet]  │  │ Name: _______________       │   │ │
│  │  │ ● walls        [Wall]    │  │ Face: [X-] [X+] [Y-]...     │   │ │
│  │  │                           │  │ Type: [Inlet ▼]            │   │ │
│  │  │ [+ Add Boundary]          │  │ Velocity: [1.0, 0, 0]      │   │ │
│  │  │                           │  │ Temperature: [293.15 K]    │   │ │
│  │  └───────────────────────────┘  └─────────────────────────────┘   │ │
│  │                                                                     │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─ Step 3: Solver Settings ──────────────────────────────────────────┐ │
│  │                                                                     │ │
│  │  Solver Type:    [○ SIMPLE  ● PISO  ○ PIMPLE]                      │ │
│  │  Turbulence:     [○ Laminar ● k-ε   ○ k-ω SST ○ LES]               │ │
│  │                                                                     │ │
│  │  Time Step (Δt):        [0.0001] s     (CFL auto-adjust)           │ │
│  │  End Time:              [1.0] s                                     │ │
│  │  Max Iterations:        [10000]                                     │ │
│  │  Convergence Tolerance: [1e-6]                                      │ │
│  │                                                                     │ │
│  │  [x] Enable thermal solver                                          │ │
│  │  [x] Enable GPU acceleration (CUDA)                                 │ │
│  │  [ ] Enable multi-GPU (requires distributed backend)                │ │
│  │                                                                     │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│                            [◀ Back]  [Start Simulation ▶]               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Results/Post-Processing Page

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Simulation: HVAC_Room_v2                    Status: ✅ Completed        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ [Velocity] [Temperature] [Pressure] [Turbulence] [Streamlines] │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                                                                     ││
│  │                                                                     ││
│  │                         3D FIELD VIEWER                             ││
│  │                         (TemperatureMap)                            ││
│  │                                                                     ││
│  │    ┌─────────────────────────────────────────────────────────────┐ ││
│  │    │ ████████████████████████████████████████████████████████ │ ││
│  │    │ 289K                    291K                        293K │ ││
│  │    └─────────────────────────────────────────────────────────────┘ ││
│  │                                                                     ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│  ┌─ Slice Controls ───────────────────────────────────────────────────┐ │
│  │                                                                     │ │
│  │  Plane: [● XY  ○ XZ  ○ YZ  ○ Custom]                               │ │
│  │                                                                     │ │
│  │  Position: ━━━━━━━━━━━●━━━━━━━━━━━ [Z = 2.5m]                       │ │
│  │                                                                     │ │
│  │  Colormap: [Viridis ▼]    Range: [Auto ▼]    [289] — [293]         │ │
│  │                                                                     │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─ Export ───────────────────────────────────────────────────────────┐ │
│  │                                                                     │ │
│  │  [📷 Screenshot PNG]  [🎥 Animation MP4]  [📊 CSV Data]  [VTK]     │ │
│  │                                                                     │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Component Hierarchy

```
App
├── RootLayout
│   ├── Providers (Query, Theme, Socket)
│   └── DashboardShell
│       ├── Sidebar
│       │   ├── Logo
│       │   ├── NavLinks
│       │   └── UserMenu
│       └── MainContent
│
├── DashboardPage
│   ├── ActiveSimulationCard
│   │   ├── MeshViewer (mini)
│   │   ├── StatusBadge
│   │   └── PerformanceMetrics
│   ├── ResidualChart
│   ├── RunControls
│   └── RecentSimulations (table)
│
├── SimulationDetailPage
│   ├── Tabs: [Setup, Running, Results]
│   ├── MeshViewer (full)
│   ├── BoundaryList
│   ├── ParameterForm
│   └── LogViewer
│
└── ResultsPage
    ├── FieldSelector
    ├── FieldViewer3D
    ├── SliceControls
    ├── ColorLegend
    └── ExportButtons
```

---

## Part 4: API Schema

### 4.1 OpenAPI Specification

```yaml
# openapi.yaml
openapi: 3.1.0
info:
  title: HyperFOAM API
  version: 1.0.0
  description: GPU-Accelerated CFD Simulation Backend

servers:
  - url: http://localhost:8000
    description: Development server
  - url: https://api.physics-os.io
    description: Production server

tags:
  - name: simulations
    description: Simulation management
  - name: meshes
    description: HyperGrid mesh operations
  - name: fields
    description: Field data retrieval
  - name: system
    description: System status

paths:
  # ==========================================
  # SIMULATIONS
  # ==========================================
  /simulations:
    get:
      tags: [simulations]
      operationId: listSimulations
      summary: List all simulations
      parameters:
        - name: status
          in: query
          schema:
            type: string
            enum: [pending, running, paused, completed, failed]
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
        - name: offset
          in: query
          schema:
            type: integer
            default: 0
      responses:
        '200':
          description: Simulation list
          content:
            application/json:
              schema:
                type: object
                properties:
                  items:
                    type: array
                    items:
                      $ref: '#/components/schemas/SimulationSummary'
                  total:
                    type: integer
                  
    post:
      tags: [simulations]
      operationId: createSimulation
      summary: Create new simulation
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SimulationCreate'
      responses:
        '201':
          description: Simulation created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Simulation'

  /simulations/{id}:
    get:
      tags: [simulations]
      operationId: getSimulation
      summary: Get simulation details
      parameters:
        - $ref: '#/components/parameters/SimulationId'
      responses:
        '200':
          description: Simulation details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Simulation'
        '404':
          $ref: '#/components/responses/NotFound'

    delete:
      tags: [simulations]
      operationId: deleteSimulation
      parameters:
        - $ref: '#/components/parameters/SimulationId'
      responses:
        '204':
          description: Deleted

  /simulations/{id}/start:
    post:
      tags: [simulations]
      operationId: startSimulation
      parameters:
        - $ref: '#/components/parameters/SimulationId'
      responses:
        '200':
          description: Simulation started
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Simulation'

  /simulations/{id}/pause:
    post:
      tags: [simulations]
      operationId: pauseSimulation
      parameters:
        - $ref: '#/components/parameters/SimulationId'
      responses:
        '200':
          description: Simulation paused

  /simulations/{id}/stop:
    post:
      tags: [simulations]
      operationId: stopSimulation
      parameters:
        - $ref: '#/components/parameters/SimulationId'
      responses:
        '200':
          description: Simulation stopped

  /simulations/{id}/residuals:
    get:
      tags: [simulations]
      operationId: getResiduals
      summary: Get convergence history
      parameters:
        - $ref: '#/components/parameters/SimulationId'
        - name: from_iteration
          in: query
          schema:
            type: integer
            default: 0
      responses:
        '200':
          description: Residual history
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/ResidualPoint'

  # ==========================================
  # MESHES
  # ==========================================
  /meshes:
    get:
      tags: [meshes]
      operationId: listMeshes
      responses:
        '200':
          description: Mesh library
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/MeshSummary'

    post:
      tags: [meshes]
      operationId: createMesh
      summary: Create HyperGrid mesh
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/MeshCreate'
      responses:
        '201':
          description: Mesh created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Mesh'

  /meshes/{id}:
    get:
      tags: [meshes]
      operationId: getMesh
      parameters:
        - $ref: '#/components/parameters/MeshId'
      responses:
        '200':
          description: Mesh details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Mesh'

  /meshes/{id}/geometry:
    get:
      tags: [meshes]
      operationId: getMeshGeometry
      summary: Get mesh geometry tensor (binary)
      parameters:
        - $ref: '#/components/parameters/MeshId'
        - name: channel
          in: query
          schema:
            type: string
            enum: [vol_frac, area_x, area_y, area_z, sdf, all]
            default: all
      responses:
        '200':
          description: Geometry tensor data
          content:
            application/octet-stream:
              schema:
                type: string
                format: binary

  /meshes/{id}/patches:
    get:
      tags: [meshes]
      operationId: getMeshPatches
      parameters:
        - $ref: '#/components/parameters/MeshId'
      responses:
        '200':
          description: Boundary patches
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/BoundaryPatch'

    post:
      tags: [meshes]
      operationId: addMeshPatch
      parameters:
        - $ref: '#/components/parameters/MeshId'
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/BoundaryPatchCreate'
      responses:
        '201':
          description: Patch added

  /meshes/upload:
    post:
      tags: [meshes]
      operationId: uploadMeshFile
      summary: Upload STL/OBJ file
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                file:
                  type: string
                  format: binary
                name:
                  type: string
                resolution:
                  type: array
                  items:
                    type: integer
                  minItems: 3
                  maxItems: 3
      responses:
        '201':
          description: Mesh created from file
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Mesh'

  # ==========================================
  # FIELDS
  # ==========================================
  /simulations/{id}/fields/{field}:
    get:
      tags: [fields]
      operationId: getFieldData
      summary: Get field tensor data
      parameters:
        - $ref: '#/components/parameters/SimulationId'
        - name: field
          in: path
          required: true
          schema:
            type: string
            enum: [velocity, pressure, temperature, turbulent_ke, turbulent_omega]
        - name: timestep
          in: query
          schema:
            type: number
            description: Simulation time (latest if omitted)
        - name: slice
          in: query
          schema:
            type: string
            description: Slice spec, e.g., "z=2.5" or "xy:32"
      responses:
        '200':
          description: Field data
          content:
            application/octet-stream:
              schema:
                type: string
                format: binary
            application/json:
              schema:
                $ref: '#/components/schemas/FieldMetadata'

  /simulations/{id}/fields/{field}/stats:
    get:
      tags: [fields]
      operationId: getFieldStats
      parameters:
        - $ref: '#/components/parameters/SimulationId'
        - name: field
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Field statistics
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/FieldStats'

  # ==========================================
  # SYSTEM
  # ==========================================
  /system/status:
    get:
      tags: [system]
      operationId: getSystemStatus
      responses:
        '200':
          description: System status
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SystemStatus'

  /system/gpus:
    get:
      tags: [system]
      operationId: listGPUs
      responses:
        '200':
          description: Available GPUs
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/GPUInfo'

# ==========================================
# COMPONENTS
# ==========================================
components:
  parameters:
    SimulationId:
      name: id
      in: path
      required: true
      schema:
        type: string
        format: uuid

    MeshId:
      name: id
      in: path
      required: true
      schema:
        type: string
        format: uuid

  responses:
    NotFound:
      description: Resource not found
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'

  schemas:
    # ---------- Simulation ----------
    SimulationSummary:
      type: object
      properties:
        id:
          type: string
          format: uuid
        name:
          type: string
        status:
          type: string
          enum: [pending, running, paused, completed, failed]
        iteration:
          type: integer
        max_iterations:
          type: integer
        created_at:
          type: string
          format: date-time
        mesh_id:
          type: string
          format: uuid

    Simulation:
      allOf:
        - $ref: '#/components/schemas/SimulationSummary'
        - type: object
          properties:
            settings:
              $ref: '#/components/schemas/SolverSettings'
            performance:
              $ref: '#/components/schemas/PerformanceMetrics'
            current_time:
              type: number
            end_time:
              type: number

    SimulationCreate:
      type: object
      required: [name, mesh_id, settings]
      properties:
        name:
          type: string
        mesh_id:
          type: string
          format: uuid
        settings:
          $ref: '#/components/schemas/SolverSettings'

    SolverSettings:
      type: object
      properties:
        solver_type:
          type: string
          enum: [SIMPLE, PISO, PIMPLE]
          default: PISO
        turbulence_model:
          type: string
          enum: [laminar, k_epsilon, k_omega_sst, les_smagorinsky]
          default: k_epsilon
        dt:
          type: number
          description: Time step (seconds)
          default: 0.0001
        end_time:
          type: number
          default: 1.0
        max_iterations:
          type: integer
          default: 10000
        convergence_tolerance:
          type: number
          default: 1e-6
        enable_thermal:
          type: boolean
          default: false
        use_gpu:
          type: boolean
          default: true

    PerformanceMetrics:
      type: object
      properties:
        cells_per_second:
          type: number
          description: Mcells/s
        gpu_utilization:
          type: number
          description: Percentage
        vram_used_gb:
          type: number
        vram_total_gb:
          type: number
        wall_time_seconds:
          type: number

    ResidualPoint:
      type: object
      properties:
        iteration:
          type: integer
        continuity:
          type: number
        momentum_x:
          type: number
        momentum_y:
          type: number
        momentum_z:
          type: number
        energy:
          type: number

    # ---------- Mesh ----------
    MeshSummary:
      type: object
      properties:
        id:
          type: string
          format: uuid
        name:
          type: string
        resolution:
          type: array
          items:
            type: integer
          minItems: 3
          maxItems: 3
        domain_size:
          type: array
          items:
            type: number
          minItems: 3
          maxItems: 3
        cell_count:
          type: integer
        created_at:
          type: string
          format: date-time

    Mesh:
      allOf:
        - $ref: '#/components/schemas/MeshSummary'
        - type: object
          properties:
            patches:
              type: array
              items:
                $ref: '#/components/schemas/BoundaryPatch'
            geometry_channels:
              type: array
              items:
                type: string
              example: [vol_frac, area_x, area_y, area_z, sdf]

    MeshCreate:
      type: object
      required: [name, resolution, domain_size]
      properties:
        name:
          type: string
        resolution:
          type: array
          items:
            type: integer
          minItems: 3
          maxItems: 3
          example: [128, 64, 64]
        domain_size:
          type: array
          items:
            type: number
          minItems: 3
          maxItems: 3
          example: [10.0, 5.0, 5.0]

    BoundaryPatch:
      type: object
      properties:
        name:
          type: string
        patch_type:
          type: string
          enum: [inlet, outlet, wall, symmetry]
        face:
          type: string
          enum: ['x-', 'x+', 'y-', 'y+', 'z-', 'z+']
        i_range:
          type: array
          items:
            type: integer
          minItems: 2
          maxItems: 2
        j_range:
          type: array
          items:
            type: integer
          minItems: 2
          maxItems: 2
        k_range:
          type: array
          items:
            type: integer
          minItems: 2
          maxItems: 2
        velocity:
          type: array
          items:
            type: number
          minItems: 3
          maxItems: 3
        temperature:
          type: number

    BoundaryPatchCreate:
      allOf:
        - $ref: '#/components/schemas/BoundaryPatch'
      required: [name, patch_type, face]

    # ---------- Fields ----------
    FieldMetadata:
      type: object
      properties:
        field:
          type: string
        shape:
          type: array
          items:
            type: integer
        dtype:
          type: string
          enum: [float32, float64]
        timestep:
          type: number
        min:
          type: number
        max:
          type: number

    FieldStats:
      type: object
      properties:
        field:
          type: string
        min:
          type: number
        max:
          type: number
        mean:
          type: number
        std:
          type: number
        timestep:
          type: number

    # ---------- System ----------
    SystemStatus:
      type: object
      properties:
        status:
          type: string
          enum: [healthy, degraded, unhealthy]
        version:
          type: string
        uptime_seconds:
          type: number
        active_simulations:
          type: integer
        queued_simulations:
          type: integer

    GPUInfo:
      type: object
      properties:
        index:
          type: integer
        name:
          type: string
        memory_total_gb:
          type: number
        memory_used_gb:
          type: number
        utilization:
          type: number
        temperature:
          type: number

    # ---------- Common ----------
    Error:
      type: object
      properties:
        code:
          type: string
        message:
          type: string
        request_id:
          type: string
```

### 4.2 WebSocket Events

```yaml
# WebSocket channels (ws://localhost:8001)

channels:
  simulation.{id}.status:
    description: Real-time simulation status updates
    payload:
      type: object
      properties:
        status:
          type: string
          enum: [running, paused, completed, failed]
        iteration:
          type: integer
        current_time:
          type: number
        performance:
          $ref: '#/components/schemas/PerformanceMetrics'

  simulation.{id}.residuals:
    description: Real-time residual updates (every N iterations)
    payload:
      $ref: '#/components/schemas/ResidualPoint'

  simulation.{id}.field:
    description: Field data snapshot (on demand or periodic)
    payload:
      type: object
      properties:
        field:
          type: string
        timestep:
          type: number
        data_url:
          type: string
          description: URL to fetch binary field data

  system.gpus:
    description: GPU utilization updates (every 5s)
    payload:
      type: array
      items:
        $ref: '#/components/schemas/GPUInfo'
```

### 4.3 Generated TypeScript Types

After running `npm run generate-api`:

```typescript
// src/lib/api/schema.d.ts (auto-generated excerpt)

export interface paths {
  "/simulations": {
    get: operations["listSimulations"];
    post: operations["createSimulation"];
  };
  "/simulations/{id}": {
    get: operations["getSimulation"];
    delete: operations["deleteSimulation"];
  };
  "/simulations/{id}/start": {
    post: operations["startSimulation"];
  };
  // ... etc
}

export interface components {
  schemas: {
    Simulation: {
      id: string;
      name: string;
      status: "pending" | "running" | "paused" | "completed" | "failed";
      iteration: number;
      max_iterations: number;
      settings: components["schemas"]["SolverSettings"];
      performance?: components["schemas"]["PerformanceMetrics"];
      current_time: number;
      end_time: number;
    };
    // ... etc
  };
}
```

---

## Summary

| Section | Status | Description |
|---------|--------|-------------|
| **1. Project Structure** | ✅ Defined | Full directory layout, package.json |
| **2. Missing Components** | ✅ Defined | shadcn/ui additions + CFD components |
| **3. Dashboard Design** | ✅ Designed | ASCII wireframes, component hierarchy |
| **4. API Schema** | ✅ Complete | OpenAPI 3.1 spec, WebSocket events |

### Next Steps

1. **Initialize project:** `cp -r creme-stack ontic-ui`
2. **Add shadcn/ui components:** Run the `npx shadcn-ui add` commands
3. **Install Three.js:** `npm install @react-three/fiber @react-three/drei three`
4. **Install Recharts:** `npm install recharts`
5. **Generate API types:** `npm run generate-api`
6. **Build CFD components:** MeshViewer, ResidualChart, BoundaryEditor
7. **Implement dashboard pages:** Dashboard, Simulations, Results

---

**Document Status:** Ready for Implementation

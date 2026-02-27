/**
 * New Simulation Wizard
 * 
 * Multi-step form for creating CFD simulations:
 * 1. Mesh Selection - Choose or upload geometry
 * 2. Boundary Conditions - Configure inlet/outlet/walls
 * 3. Solver Settings - Turbulence model, time stepping, convergence
 * 
 * @article III - Comprehensive validation with Zod
 * @article VIII - Error handling with graceful recovery
 */

'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import {
  ArrowLeft,
  ArrowRight,
  Box,
  Settings2,
  Layers,
  Play,
  Loader2,
  CheckCircle2,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Sidebar,
  Header,
  DashboardShell,
  PageHeader,
} from '@/components/layout';
import { MeshViewer } from '@/components/cfd';
import { IntakeForm, type IntakeFormData } from '@/components/intake';
import { useMeshes, useMesh, useCreateSimulation } from '@/hooks';
import type { SolverType, TurbulenceModel } from '@/types';

// ============================================
// FORM SCHEMA
// ============================================

const simulationSchema = z.object({
  name: z.string().min(1, 'Name is required').max(100),
  description: z.string().max(500).optional(),
  meshId: z.string().uuid('Please select a mesh'),
  solverType: z.enum(['steady', 'transient', 'pseudo-transient']),
  turbulenceModel: z.enum(['laminar', 'ke-standard', 'ke-realizable', 'kw-sst', 'sa', 'les-smagorinsky', 'des']),
  dt: z.number().min(1e-8).max(1),
  endTime: z.number().min(0).max(1000),
  maxIterations: z.number().int().min(1).max(1000000),
  convergenceTolerance: z.number().min(1e-12).max(1),
  enableThermal: z.boolean(),
  useGpu: z.boolean(),
  cflNumber: z.number().min(0.1).max(10),
});

type SimulationFormData = z.infer<typeof simulationSchema>;

// ============================================
// STEPS
// ============================================

const STEPS = [
  { id: 'mesh', label: 'Geometry', icon: Box },
  { id: 'intake', label: 'Project Setup', icon: Layers },
  { id: 'solver', label: 'Solver', icon: Settings2 },
] as const;

type StepId = typeof STEPS[number]['id'];

// ============================================
// MAIN COMPONENT
// ============================================

export default function NewSimulationPage() {
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [intakeData, setIntakeData] = useState<IntakeFormData | null>(null);
  
  const { data: meshes, isLoading: meshesLoading } = useMeshes();
  const createSimulation = useCreateSimulation();

  const form = useForm<SimulationFormData>({
    resolver: zodResolver(simulationSchema),
    defaultValues: {
      name: '',
      description: '',
      meshId: '',
      solverType: 'steady',
      turbulenceModel: 'ke-standard',
      dt: 0.0001,
      endTime: 1.0,
      maxIterations: 10000,
      convergenceTolerance: 1e-6,
      enableThermal: false,
      useGpu: true,
      cflNumber: 0.9,
    },
  });

  const selectedMeshId = form.watch('meshId');
  const { data: selectedMesh } = useMesh(selectedMeshId || null);

  const handleNext = () => {
    if (currentStep < STEPS.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSubmit = async (data: SimulationFormData) => {
    try {
      const result = await createSimulation.mutateAsync({
        name: data.name,
        description: data.description,
        meshId: data.meshId,
        settings: {
          solverType: data.solverType,
          turbulenceModel: data.turbulenceModel,
          maxIterations: data.maxIterations,
          convergenceTolerance: data.convergenceTolerance,
          cflNumber: data.cflNumber,
          gpuAcceleration: data.useGpu,
          timeStep: data.dt,
          endTime: data.endTime,
        },
      });
      if (result) {
        router.push(`/simulations/${result.id}`);
      }
    } catch (error) {
      console.error('Failed to create simulation:', error);
    }
  };

  const progress = ((currentStep + 1) / STEPS.length) * 100;

  return (
    <DashboardShell>
      <div className="container max-w-4xl py-6">
        {/* Header */}
        <div className="mb-8">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => router.back()}
            className="mb-4"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Simulations
          </Button>
          
          <h1 className="text-3xl font-bold">New Simulation</h1>
          <p className="text-muted-foreground mt-1">
            Configure your CFD simulation in three easy steps
          </p>
        </div>

        {/* Progress */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-2">
            {STEPS.map((step, index) => {
              const Icon = step.icon;
              const isActive = index === currentStep;
              const isCompleted = index < currentStep;
              
              return (
                <div
                  key={step.id}
                  className={`flex items-center gap-2 ${
                    isActive
                      ? 'text-primary'
                      : isCompleted
                      ? 'text-muted-foreground'
                      : 'text-muted-foreground/50'
                  }`}
                >
                  <div
                    className={`h-8 w-8 rounded-full flex items-center justify-center ${
                      isActive
                        ? 'bg-primary text-primary-foreground'
                        : isCompleted
                        ? 'bg-primary/20 text-primary'
                        : 'bg-muted'
                    }`}
                  >
                    {isCompleted ? (
                      <CheckCircle2 className="h-4 w-4" />
                    ) : (
                      <Icon className="h-4 w-4" />
                    )}
                  </div>
                  <span className="hidden sm:inline font-medium">
                    {step.label}
                  </span>
                </div>
              );
            })}
          </div>
          <Progress value={progress} className="h-2" />
        </div>

        {/* Form */}
        <form onSubmit={form.handleSubmit(handleSubmit)}>
          {/* Step 1: Mesh Selection */}
          {currentStep === 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Select Geometry</CardTitle>
                <CardDescription>
                  Choose a mesh from your library or upload a new one
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Simulation Name */}
                <div className="space-y-2">
                  <Label htmlFor="name">Simulation Name</Label>
                  <Input
                    id="name"
                    placeholder="e.g., HVAC Room Flow Analysis"
                    {...form.register('name')}
                  />
                  {form.formState.errors.name && (
                    <p className="text-sm text-destructive">
                      {form.formState.errors.name.message}
                    </p>
                  )}
                </div>

                {/* Description */}
                <div className="space-y-2">
                  <Label htmlFor="description">Description (optional)</Label>
                  <Input
                    id="description"
                    placeholder="Brief description of the simulation"
                    {...form.register('description')}
                  />
                </div>

                <Separator />

                {/* Mesh Selection */}
                <div className="space-y-2">
                  <Label>Select Mesh</Label>
                  <Select
                    value={selectedMeshId}
                    onValueChange={(v) => form.setValue('meshId', v)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Choose a mesh..." />
                    </SelectTrigger>
                    <SelectContent>
                      {meshesLoading ? (
                        <div className="px-2 py-1.5 text-sm text-muted-foreground">
                          Loading meshes...
                        </div>
                      ) : meshes?.length === 0 ? (
                        <div className="px-2 py-1.5 text-sm text-muted-foreground">
                          No meshes available
                        </div>
                      ) : (
                        meshes?.map((mesh) => (
                          <SelectItem key={mesh.id} value={mesh.id}>
                            {mesh.name} ({mesh.cellCount.toLocaleString()} cells)
                          </SelectItem>
                        ))
                      )}
                    </SelectContent>
                  </Select>
                  {form.formState.errors.meshId && (
                    <p className="text-sm text-destructive">
                      {form.formState.errors.meshId.message}
                    </p>
                  )}
                </div>

                {/* Mesh Preview */}
                <div className="h-64 rounded-lg border overflow-hidden">
                  <MeshViewer
                    mesh={selectedMesh}
                    isLoading={meshesLoading}
                    className="h-full"
                  />
                </div>

                {/* Mesh Info */}
                {selectedMesh && (
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Resolution</span>
                      <p className="font-mono">
                        {selectedMesh.resolution.nx}×{selectedMesh.resolution.ny}×{selectedMesh.resolution.nz}
                      </p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Cell Count</span>
                      <p className="font-mono">
                        {selectedMesh.cellCount?.toLocaleString() ?? 'N/A'}
                      </p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Domain Size</span>
                      <p className="font-mono">
                        {selectedMesh.domain_size?.join('×') ?? 'N/A'} m
                      </p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Step 2: Project Intake Form */}
          {currentStep === 1 && (
            <Card>
              <CardHeader>
                <CardTitle>Project Setup</CardTitle>
                <CardDescription>
                  Configure your HVAC simulation parameters
                </CardDescription>
              </CardHeader>
              <CardContent>
                <IntakeForm
                  onSubmit={(data) => {
                    setIntakeData(data);
                    handleNext();
                  }}
                  initialData={intakeData ?? undefined}
                />
              </CardContent>
            </Card>
          )}

          {/* Step 3: Solver Settings */}
          {currentStep === 2 && (
            <Card>
              <CardHeader>
                <CardTitle>Solver Settings</CardTitle>
                <CardDescription>
                  Configure numerical solver parameters
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid gap-6 md:grid-cols-2">
                  {/* Solver Type */}
                  <div className="space-y-2">
                    <Label>Solver Type</Label>
                    <Select
                      value={form.watch('solverType')}
                      onValueChange={(v) =>
                        form.setValue('solverType', v as SolverType)
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="steady">Steady-State</SelectItem>
                        <SelectItem value="transient">Transient</SelectItem>
                        <SelectItem value="pseudo-transient">
                          Pseudo-Transient
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Turbulence Model */}
                  <div className="space-y-2">
                    <Label>Turbulence Model</Label>
                    <Select
                      value={form.watch('turbulenceModel')}
                      onValueChange={(v) =>
                        form.setValue('turbulenceModel', v as TurbulenceModel)
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="laminar">Laminar</SelectItem>
                        <SelectItem value="ke-standard">k-ε Standard</SelectItem>
                        <SelectItem value="ke-realizable">k-ε Realizable</SelectItem>
                        <SelectItem value="kw-sst">k-ω SST</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Time Step */}
                  <div className="space-y-2">
                    <Label htmlFor="dt">Time Step (s)</Label>
                    <Input
                      id="dt"
                      type="number"
                      step="0.0001"
                      {...form.register('dt', { valueAsNumber: true })}
                    />
                  </div>

                  {/* End Time */}
                  <div className="space-y-2">
                    <Label htmlFor="endTime">End Time (s)</Label>
                    <Input
                      id="endTime"
                      type="number"
                      step="0.1"
                      {...form.register('endTime', { valueAsNumber: true })}
                    />
                  </div>

                  {/* Max Iterations */}
                  <div className="space-y-2">
                    <Label htmlFor="maxIterations">Max Iterations</Label>
                    <Input
                      id="maxIterations"
                      type="number"
                      {...form.register('maxIterations', { valueAsNumber: true })}
                    />
                  </div>

                  {/* Convergence Tolerance */}
                  <div className="space-y-2">
                    <Label htmlFor="convergenceTolerance">
                      Convergence Tolerance
                    </Label>
                    <Input
                      id="convergenceTolerance"
                      type="number"
                      step="1e-7"
                      {...form.register('convergenceTolerance', {
                        valueAsNumber: true,
                      })}
                    />
                  </div>
                </div>

                <Separator />

                {/* CFL Number */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label>CFL Number</Label>
                    <span className="font-mono text-sm">
                      {form.watch('cflNumber').toFixed(1)}
                    </span>
                  </div>
                  <Slider
                    value={[form.watch('cflNumber')]}
                    onValueChange={(values: number[]) => values[0] !== undefined && form.setValue('cflNumber', values[0])}
                    min={0.1}
                    max={2.0}
                    step={0.1}
                  />
                </div>

                <Separator />

                {/* Options */}
                <div className="space-y-4">
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="enableThermal"
                      checked={form.watch('enableThermal')}
                      onCheckedChange={(v) =>
                        form.setValue('enableThermal', !!v)
                      }
                    />
                    <Label htmlFor="enableThermal">
                      Enable thermal solver (energy equation)
                    </Label>
                  </div>

                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="useGpu"
                      checked={form.watch('useGpu')}
                      onCheckedChange={(v) => form.setValue('useGpu', !!v)}
                    />
                    <Label htmlFor="useGpu">
                      Enable GPU acceleration (CUDA)
                    </Label>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Navigation */}
          <div className="flex items-center justify-between mt-6">
            <Button
              type="button"
              variant="outline"
              onClick={handleBack}
              disabled={currentStep === 0}
            >
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>

            {currentStep < STEPS.length - 1 ? (
              <Button type="button" onClick={handleNext}>
                Next
                <ArrowRight className="h-4 w-4 ml-2" />
              </Button>
            ) : (
              <Button
                type="submit"
                disabled={createSimulation.isPending}
              >
                {createSimulation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Creating...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Create Simulation
                  </>
                )}
              </Button>
            )}
          </div>
        </form>
      </div>
    </DashboardShell>
  );
}

/**
 * ParameterForm - Solver Settings Configuration
 * 
 * Form for configuring simulation solver parameters with validation.
 */

'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Settings, Zap, Flame, Wind, Timer, Cpu } from 'lucide-react';
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
import { Switch } from '@/components/ui/switch';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { type SolverSettings, DEFAULT_SOLVER_SETTINGS } from '@/types';

// ============================================
// VALIDATION SCHEMA (camelCase to match types)
// ============================================

const solverSettingsSchema = z.object({
  solverType: z.enum(['steady', 'transient', 'pseudo-transient']),
  turbulenceModel: z.enum(['laminar', 'ke-standard', 'ke-realizable', 'kw-sst', 'sa', 'les-smagorinsky', 'des']),
  maxIterations: z.number().min(1).max(100000),
  convergenceTolerance: z.number().min(1e-12).max(1e-1),
  cflNumber: z.number().min(0.1).max(100),
  timeStep: z.number().min(1e-10).max(1).optional(),
  endTime: z.number().min(0).optional(),
  relaxationFactors: z.object({
    pressure: z.number().min(0.01).max(1),
    velocity: z.number().min(0.01).max(1),
    energy: z.number().min(0.01).max(1),
    turbulence: z.number().min(0.01).max(1),
  }),
  discretization: z.object({
    convection: z.enum(['upwind', 'central', 'quick', 'muscl', 'weno']),
    gradient: z.enum(['green-gauss', 'least-squares', 'weighted-least-squares']),
    time: z.enum(['euler-implicit', 'crank-nicolson', 'bdf2', 'rk4']).optional(),
  }),
  gpuAcceleration: z.boolean(),
  precision: z.enum(['fp32', 'fp64', 'mixed']),
});

type SolverFormData = z.infer<typeof solverSettingsSchema>;

// ============================================
// PROPS INTERFACE
// ============================================

interface ParameterFormProps {
  initialValues?: Partial<SolverSettings>;
  onSubmit: (settings: SolverSettings) => void;
  isLoading?: boolean;
  className?: string;
}

// ============================================
// MAIN COMPONENT
// ============================================

export function ParameterForm({
  initialValues,
  onSubmit,
  isLoading = false,
  className = '',
}: ParameterFormProps) {
  const {
    register,
    handleSubmit,
    watch,
    setValue,
    formState: { errors, isDirty },
  } = useForm<SolverFormData>({
    resolver: zodResolver(solverSettingsSchema),
    defaultValues: {
      ...DEFAULT_SOLVER_SETTINGS,
      ...initialValues,
    } as SolverFormData,
  });

  const solverType = watch('solverType');
  const isTransient = solverType === 'transient';

  const handleFormSubmit = (data: SolverFormData) => {
    onSubmit(data as SolverSettings);
  };

  return (
    <TooltipProvider delayDuration={0}>
      <form onSubmit={handleSubmit(handleFormSubmit)} className={className}>
        <Accordion type="multiple" defaultValue={['general', 'numerics']} className="space-y-4">
          {/* General Settings */}
          <AccordionItem value="general" className="border rounded-lg px-4">
            <AccordionTrigger className="hover:no-underline">
              <div className="flex items-center gap-2">
                <Settings className="h-4 w-4" />
                <span>General Settings</span>
              </div>
            </AccordionTrigger>
            <AccordionContent className="space-y-4 pt-4">
              {/* Solver Type */}
              <div className="space-y-2">
                <Label>Solver Type</Label>
                <Select
                  value={watch('solverType')}
                  onValueChange={(v) => setValue('solverType', v as SolverFormData['solverType'])}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="steady">Steady-State</SelectItem>
                    <SelectItem value="transient">Transient</SelectItem>
                    <SelectItem value="pseudo-transient">Pseudo-Transient</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Turbulence Model */}
              <div className="space-y-2">
                <Label className="flex items-center gap-2">
                  <Wind className="h-4 w-4" />
                  Turbulence Model
                </Label>
                <Select
                  value={watch('turbulenceModel')}
                  onValueChange={(v) => setValue('turbulenceModel', v as SolverFormData['turbulenceModel'])}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="laminar">Laminar</SelectItem>
                    <SelectItem value="ke-standard">k-ε Standard</SelectItem>
                    <SelectItem value="ke-realizable">k-ε Realizable</SelectItem>
                    <SelectItem value="kw-sst">k-ω SST</SelectItem>
                    <SelectItem value="sa">Spalart-Allmaras</SelectItem>
                    <SelectItem value="les-smagorinsky">LES Smagorinsky</SelectItem>
                    <SelectItem value="des">DES</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Max Iterations */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Label className="flex items-center gap-2 cursor-help">
                        <Timer className="h-4 w-4" />
                        Max Iterations
                      </Label>
                    </TooltipTrigger>
                    <TooltipContent>
                      Maximum number of solver iterations
                    </TooltipContent>
                  </Tooltip>
                  <Input
                    type="number"
                    {...register('maxIterations', { valueAsNumber: true })}
                  />
                  {errors.maxIterations && (
                    <p className="text-xs text-destructive">{errors.maxIterations.message}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Label className="cursor-help">Convergence Tolerance</Label>
                    </TooltipTrigger>
                    <TooltipContent>
                      Residual threshold for convergence (e.g., 1e-6)
                    </TooltipContent>
                  </Tooltip>
                  <Input
                    type="number"
                    step="any"
                    {...register('convergenceTolerance', { valueAsNumber: true })}
                  />
                </div>
              </div>

              {/* Transient Settings */}
              {isTransient && (
                <Card className="bg-muted/50">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Transient Settings</CardTitle>
                  </CardHeader>
                  <CardContent className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Time Step (s)</Label>
                      <Input
                        type="number"
                        step="any"
                        {...register('timeStep', { valueAsNumber: true })}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>End Time (s)</Label>
                      <Input
                        type="number"
                        step="any"
                        {...register('endTime', { valueAsNumber: true })}
                      />
                    </div>
                  </CardContent>
                </Card>
              )}
            </AccordionContent>
          </AccordionItem>

          {/* Numerical Settings */}
          <AccordionItem value="numerics" className="border rounded-lg px-4">
            <AccordionTrigger className="hover:no-underline">
              <div className="flex items-center gap-2">
                <Zap className="h-4 w-4" />
                <span>Numerical Schemes</span>
              </div>
            </AccordionTrigger>
            <AccordionContent className="space-y-4 pt-4">
              {/* CFL Number */}
              <div className="space-y-2">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Label className="cursor-help">CFL Number</Label>
                  </TooltipTrigger>
                  <TooltipContent>
                    Courant-Friedrichs-Lewy number for stability
                  </TooltipContent>
                </Tooltip>
                <Input
                  type="number"
                  step="0.1"
                  {...register('cflNumber', { valueAsNumber: true })}
                />
              </div>

              {/* Discretization Schemes */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Convection Scheme</Label>
                  <Select
                    value={watch('discretization.convection')}
                    onValueChange={(v) => setValue('discretization.convection', v as any)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="upwind">Upwind (1st order)</SelectItem>
                      <SelectItem value="central">Central</SelectItem>
                      <SelectItem value="quick">QUICK</SelectItem>
                      <SelectItem value="muscl">MUSCL (2nd order)</SelectItem>
                      <SelectItem value="weno">WENO (5th order)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Gradient Scheme</Label>
                  <Select
                    value={watch('discretization.gradient')}
                    onValueChange={(v) => setValue('discretization.gradient', v as any)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="green-gauss">Green-Gauss</SelectItem>
                      <SelectItem value="least-squares">Least Squares</SelectItem>
                      <SelectItem value="weighted-least-squares">Weighted LS</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* Relaxation Factors */}
              <div className="space-y-3">
                <Label>Relaxation Factors</Label>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">Pressure</Label>
                    <Input
                      type="number"
                      step="0.05"
                      {...register('relaxationFactors.pressure', { valueAsNumber: true })}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">Velocity</Label>
                    <Input
                      type="number"
                      step="0.05"
                      {...register('relaxationFactors.velocity', { valueAsNumber: true })}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">Energy</Label>
                    <Input
                      type="number"
                      step="0.05"
                      {...register('relaxationFactors.energy', { valueAsNumber: true })}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">Turbulence</Label>
                    <Input
                      type="number"
                      step="0.05"
                      {...register('relaxationFactors.turbulence', { valueAsNumber: true })}
                    />
                  </div>
                </div>
              </div>
            </AccordionContent>
          </AccordionItem>

          {/* Performance Settings */}
          <AccordionItem value="performance" className="border rounded-lg px-4">
            <AccordionTrigger className="hover:no-underline">
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4" />
                <span>Performance</span>
              </div>
            </AccordionTrigger>
            <AccordionContent className="space-y-4 pt-4">
              {/* GPU Acceleration */}
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label className="flex items-center gap-2">
                    <Flame className="h-4 w-4 text-orange-500" />
                    GPU Acceleration
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    Enable CUDA acceleration for HyperGrid solver
                  </p>
                </div>
                <Switch
                  checked={watch('gpuAcceleration')}
                  onCheckedChange={(checked) => setValue('gpuAcceleration', checked)}
                />
              </div>

              {/* Precision */}
              <div className="space-y-2">
                <Label>Floating Point Precision</Label>
                <Select
                  value={watch('precision')}
                  onValueChange={(v) => setValue('precision', v as SolverFormData['precision'])}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="fp32">Single (FP32)</SelectItem>
                    <SelectItem value="fp64">Double (FP64)</SelectItem>
                    <SelectItem value="mixed">Mixed Precision</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  Mixed precision uses FP32 for most operations, FP64 for residuals
                </p>
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>

        {/* Submit Button */}
        <div className="flex justify-end mt-6">
          <Button type="submit" disabled={isLoading || !isDirty}>
            {isLoading ? 'Applying...' : 'Apply Settings'}
          </Button>
        </div>
      </form>
    </TooltipProvider>
  );
}

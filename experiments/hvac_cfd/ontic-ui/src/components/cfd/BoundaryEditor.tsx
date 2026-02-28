/**
 * BoundaryEditor - Boundary Condition Configuration
 * 
 * Form component for adding/editing boundary conditions on HyperGrid patches.
 * Supports inlet, outlet, wall, and symmetry boundary types.
 */

'use client';

import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { 
  Select, 
  SelectTrigger, 
  SelectValue, 
  SelectContent, 
  SelectItem 
} from '@/components/ui/select';
import type { BoundaryPatchCreate, PatchType, FaceDirection } from '@/types';

// ============================================
// VALIDATION SCHEMA
// ============================================

const boundarySchema = z.object({
  name: z.string().min(1, 'Name is required').max(50),
  face: z.enum(['x-', 'x+', 'y-', 'y+', 'z-', 'z+'] as const),
  patch_type: z.enum(['inlet', 'outlet', 'wall', 'symmetry'] as const),
  velocity_x: z.number().optional(),
  velocity_y: z.number().optional(),
  velocity_z: z.number().optional(),
  temperature: z.number().min(0).optional(),
});

export type BoundaryFormData = z.infer<typeof boundarySchema>;

// ============================================
// PROPS INTERFACE
// ============================================

interface BoundaryEditorProps {
  onSubmit: (data: BoundaryPatchCreate) => void;
  onCancel?: () => void;
  initialData?: Partial<BoundaryPatchCreate>;
  isLoading?: boolean;
  className?: string;
}

// ============================================
// FACE OPTIONS
// ============================================

const FACE_OPTIONS: { value: FaceDirection; label: string }[] = [
  { value: 'x-', label: 'X− (Left)' },
  { value: 'x+', label: 'X+ (Right)' },
  { value: 'y-', label: 'Y− (Front)' },
  { value: 'y+', label: 'Y+ (Back)' },
  { value: 'z-', label: 'Z− (Bottom)' },
  { value: 'z+', label: 'Z+ (Top)' },
];

const TYPE_OPTIONS: { value: PatchType; label: string; description: string }[] = [
  { value: 'inlet', label: 'Inlet', description: 'Fixed velocity (Dirichlet)' },
  { value: 'outlet', label: 'Outlet', description: 'Zero gradient (Neumann)' },
  { value: 'wall', label: 'Wall', description: 'No-slip condition' },
  { value: 'symmetry', label: 'Symmetry', description: 'Mirror boundary' },
];

// ============================================
// MAIN COMPONENT
// ============================================

export function BoundaryEditor({
  onSubmit,
  onCancel,
  initialData,
  isLoading = false,
  className = '',
}: BoundaryEditorProps) {
  const {
    register,
    handleSubmit,
    control,
    watch,
    formState: { errors },
  } = useForm<BoundaryFormData>({
    resolver: zodResolver(boundarySchema),
    defaultValues: {
      name: initialData?.name ?? '',
      face: initialData?.face ?? 'x-',
      patch_type: (initialData?.patch_type as 'inlet' | 'outlet' | 'wall' | 'symmetry') ?? 'wall',
      velocity_x: initialData?.velocity?.[0] ?? 0,
      velocity_y: initialData?.velocity?.[1] ?? 0,
      velocity_z: initialData?.velocity?.[2] ?? 0,
      temperature: initialData?.temperature ?? 293.15,
    },
  });

  const patchType = watch('patch_type');
  const showVelocity = patchType === 'inlet';
  const showTemperature = patchType === 'inlet' || patchType === 'wall';

  const handleFormSubmit = (data: BoundaryFormData) => {
    const patch: BoundaryPatchCreate = {
      name: data.name,
      face: data.face,
      patch_type: data.patch_type,
    };

    if (showVelocity) {
      patch.velocity = [
        data.velocity_x ?? 0,
        data.velocity_y ?? 0,
        data.velocity_z ?? 0,
      ];
    }

    if (showTemperature && data.temperature) {
      patch.temperature = data.temperature;
    }

    onSubmit(patch);
  };

  return (
    <Card className={className}>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">
          {initialData ? 'Edit Boundary' : 'Add Boundary Condition'}
        </CardTitle>
      </CardHeader>

      <form onSubmit={handleSubmit(handleFormSubmit)}>
        <CardContent className="space-y-4">
          {/* Patch Name */}
          <div className="space-y-2">
            <Label htmlFor="name">Patch Name</Label>
            <Input
              id="name"
              placeholder="e.g., inlet_left, outlet_right"
              {...register('name')}
              className={errors.name ? 'border-destructive' : ''}
            />
            {errors.name && (
              <p className="text-xs text-destructive">{errors.name.message}</p>
            )}
          </div>

          {/* Face Selection */}
          <div className="space-y-2">
            <Label>Face</Label>
            <Controller
              name="face"
              control={control}
              render={({ field }) => (
                <Select onValueChange={field.onChange} value={field.value}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select face" />
                  </SelectTrigger>
                  <SelectContent>
                    {FACE_OPTIONS.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            />
          </div>

          {/* Boundary Type */}
          <div className="space-y-2">
            <Label>Boundary Type</Label>
            <Controller
              name="patch_type"
              control={control}
              render={({ field }) => (
                <Select onValueChange={field.onChange} value={field.value}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select type" />
                  </SelectTrigger>
                  <SelectContent>
                    {TYPE_OPTIONS.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        <div className="flex flex-col">
                          <span>{opt.label}</span>
                          <span className="text-xs text-muted-foreground">
                            {opt.description}
                          </span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            />
          </div>

          {/* Velocity (for inlet) */}
          {showVelocity && (
            <div className="space-y-2">
              <Label>Velocity (m/s)</Label>
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <Input
                    type="number"
                    step="0.01"
                    placeholder="Ux"
                    {...register('velocity_x', { valueAsNumber: true })}
                  />
                  <span className="text-xs text-muted-foreground">Ux</span>
                </div>
                <div>
                  <Input
                    type="number"
                    step="0.01"
                    placeholder="Uy"
                    {...register('velocity_y', { valueAsNumber: true })}
                  />
                  <span className="text-xs text-muted-foreground">Uy</span>
                </div>
                <div>
                  <Input
                    type="number"
                    step="0.01"
                    placeholder="Uz"
                    {...register('velocity_z', { valueAsNumber: true })}
                  />
                  <span className="text-xs text-muted-foreground">Uz</span>
                </div>
              </div>
            </div>
          )}

          {/* Temperature */}
          {showTemperature && (
            <div className="space-y-2">
              <Label htmlFor="temperature">Temperature (K)</Label>
              <Input
                id="temperature"
                type="number"
                step="0.1"
                min="0"
                placeholder="293.15"
                {...register('temperature', { valueAsNumber: true })}
              />
            </div>
          )}
        </CardContent>

        <CardFooter className="flex justify-end gap-2">
          {onCancel && (
            <Button type="button" variant="outline" onClick={onCancel}>
              Cancel
            </Button>
          )}
          <Button type="submit" disabled={isLoading}>
            {isLoading ? 'Saving...' : initialData ? 'Update' : 'Add Boundary'}
          </Button>
        </CardFooter>
      </form>
    </Card>
  );
}

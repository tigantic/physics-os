/**
 * IntakeForm - HVAC CFD Project Intake Form
 * 
 * User-friendly form for collecting HVAC simulation parameters.
 * Replaces technical "boundary conditions" with intuitive fields.
 * 
 * @article III - Comprehensive Zod validation
 * @article VI - Full documentation
 */

'use client';

import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useState } from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import {
  Building2,
  Thermometer,
  Wind,
  Users,
  Settings2,
  FileCheck,
  ChevronRight,
} from 'lucide-react';

// ============================================
// VALIDATION SCHEMA
// ============================================

const intakeSchema = z.object({
  // Project Info
  project_name: z.string().min(1, 'Project name is required').max(100),
  project_description: z.string().max(500).optional(),
  
  // Units
  unit_system: z.enum(['imperial', 'metric']),
  
  // Geometry (MANDATORY)
  room_name: z.string().min(1, 'Room name is required'),
  room_length: z.number().min(1).max(1000),
  room_width: z.number().min(1).max(1000),
  room_height: z.number().min(1).max(100),
  
  // HVAC System (MANDATORY)
  hvac_system_type: z.enum([
    'vav', 'cav', 'doas', 'vrf', 'split', 'ptac', 'radiant', 'chilled_beam', 'other'
  ]),
  vent_count: z.number().int().min(1).max(500),
  return_count: z.number().int().min(0).max(100),
  diffuser_type: z.enum([
    'ceiling_4way', 'ceiling_2way', 'ceiling_1way', 'ceiling_round', 
    'ceiling_perforated', 'sidewall', 'floor', 'displacement', 'slot', 'jet'
  ]),
  supply_airflow: z.number().min(1).max(1000000),
  supply_temperature: z.number().min(32).max(120),
  
  // Wall Boundaries (RECOMMENDED)
  wall_boundary_type: z.enum(['adiabatic', 'fixed_temp', 'u_value']),
  wall_temperature: z.number().optional(),
  
  // Thermal Loads (RECOMMENDED)
  occupancy: z.number().int().min(0).max(10000),
  occupant_activity: z.enum([
    'sleeping', 'seated_quiet', 'office', 'standing_light', 
    'walking', 'light_machine', 'heavy_work'
  ]),
  lighting_load: z.number().min(0).max(100000),
  equipment_load: z.number().min(0).max(500000),
  
  // Design Conditions
  indoor_setpoint_cooling: z.number().min(50).max(90),
  
  // Solver (OPTIONAL)
  turbulence_model: z.enum([
    'k_epsilon', 'k_epsilon_rng', 'k_omega', 'k_omega_sst', 
    'spalart_allmaras', 'les', 'laminar'
  ]),
  grid_resolution: z.enum(['coarse', 'medium', 'fine', 'very_fine']),
  steady_state: z.boolean(),
  
  // Compliance
  adpi_target: z.number().min(0).max(100),
  ventilation_standard: z.enum([
    'ashrae_62_1', 'ashrae_62_2', 'en_16798', 'gb_50736', 'custom'
  ]),
  space_type_ashrae: z.enum([
    'office', 'conference', 'classroom', 'retail', 'restaurant',
    'hotel_lobby', 'hotel_room', 'hospital_patient', 'laboratory',
    'gymnasium', 'warehouse', 'data_center'
  ]),
});

export type IntakeFormData = z.infer<typeof intakeSchema>;

// ============================================
// DEFAULT VALUES
// ============================================

const defaultValues: IntakeFormData = {
  project_name: '',
  project_description: '',
  unit_system: 'imperial',
  room_name: '',
  room_length: 30,
  room_width: 20,
  room_height: 10,
  hvac_system_type: 'vav',
  vent_count: 4,
  return_count: 1,
  diffuser_type: 'ceiling_4way',
  supply_airflow: 800,
  supply_temperature: 55,
  wall_boundary_type: 'adiabatic',
  wall_temperature: 72,
  occupancy: 10,
  occupant_activity: 'office',
  lighting_load: 500,
  equipment_load: 1000,
  indoor_setpoint_cooling: 75,
  turbulence_model: 'k_epsilon',
  grid_resolution: 'medium',
  steady_state: true,
  adpi_target: 80,
  ventilation_standard: 'ashrae_62_1',
  space_type_ashrae: 'office',
};

// ============================================
// PROPS
// ============================================

interface IntakeFormProps {
  onSubmit: (data: IntakeFormData) => void;
  onCancel?: () => void;
  initialData?: Partial<IntakeFormData>;
  isLoading?: boolean;
  className?: string;
}

// ============================================
// COMPONENT
// ============================================

export function IntakeForm({
  onSubmit,
  onCancel,
  initialData,
  isLoading = false,
  className = '',
}: IntakeFormProps) {
  const [expandedSections, setExpandedSections] = useState<string[]>([
    'project', 'geometry', 'hvac'
  ]);

  const form = useForm<IntakeFormData>({
    resolver: zodResolver(intakeSchema),
    defaultValues: { ...defaultValues, ...initialData },
  });

  const unitSystem = form.watch('unit_system');
  const lengthUnit = unitSystem === 'imperial' ? 'ft' : 'm';
  const tempUnit = unitSystem === 'imperial' ? '°F' : '°C';
  const airflowUnit = unitSystem === 'imperial' ? 'CFM' : 'm³/s';

  const handleFormSubmit = (data: IntakeFormData) => {
    onSubmit(data);
  };

  return (
    <form 
      onSubmit={form.handleSubmit(handleFormSubmit)} 
      className={className}
    >
      <Accordion
        type="multiple"
        value={expandedSections}
        onValueChange={setExpandedSections}
        className="space-y-4"
      >
        {/* ============================================ */}
        {/* PROJECT INFORMATION */}
        {/* ============================================ */}
        <AccordionItem value="project" className="border rounded-lg">
          <AccordionTrigger className="px-4 hover:no-underline">
            <div className="flex items-center gap-3">
              <Building2 className="h-5 w-5 text-primary" />
              <span className="font-semibold">Project Information</span>
              <Badge variant="outline" className="ml-2">Required</Badge>
            </div>
          </AccordionTrigger>
          <AccordionContent className="px-4 pb-4">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="project_name">Project Name *</Label>
                <Input
                  id="project_name"
                  placeholder="Office HVAC Analysis"
                  {...form.register('project_name')}
                />
                {form.formState.errors.project_name && (
                  <p className="text-sm text-destructive">
                    {form.formState.errors.project_name.message}
                  </p>
                )}
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="unit_system">Unit System</Label>
                <Controller
                  name="unit_system"
                  control={form.control}
                  render={({ field }) => (
                    <Select onValueChange={field.onChange} value={field.value}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="imperial">Imperial (ft, °F, CFM)</SelectItem>
                        <SelectItem value="metric">Metric (m, °C, m³/s)</SelectItem>
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>

              <div className="space-y-2 md:col-span-2">
                <Label htmlFor="project_description">Description</Label>
                <Input
                  id="project_description"
                  placeholder="Brief description of the project..."
                  {...form.register('project_description')}
                />
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>

        {/* ============================================ */}
        {/* ROOM GEOMETRY */}
        {/* ============================================ */}
        <AccordionItem value="geometry" className="border rounded-lg">
          <AccordionTrigger className="px-4 hover:no-underline">
            <div className="flex items-center gap-3">
              <Building2 className="h-5 w-5 text-primary" />
              <span className="font-semibold">Room Geometry</span>
              <Badge variant="outline" className="ml-2">Required</Badge>
            </div>
          </AccordionTrigger>
          <AccordionContent className="px-4 pb-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <div className="space-y-2 md:col-span-2">
                <Label htmlFor="room_name">Room/Zone Name *</Label>
                <Input
                  id="room_name"
                  placeholder="Main Office Area"
                  {...form.register('room_name')}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="room_length">Length ({lengthUnit}) *</Label>
                <Input
                  id="room_length"
                  type="number"
                  step="0.1"
                  {...form.register('room_length', { valueAsNumber: true })}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="room_width">Width ({lengthUnit}) *</Label>
                <Input
                  id="room_width"
                  type="number"
                  step="0.1"
                  {...form.register('room_width', { valueAsNumber: true })}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="room_height">Ceiling Height ({lengthUnit}) *</Label>
                <Input
                  id="room_height"
                  type="number"
                  step="0.1"
                  {...form.register('room_height', { valueAsNumber: true })}
                />
              </div>

              <div className="space-y-2">
                <Label>Calculated Volume</Label>
                <div className="p-2 bg-muted rounded-md text-sm">
                  {(form.watch('room_length') * form.watch('room_width') * form.watch('room_height')).toFixed(0)} {lengthUnit}³
                </div>
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>

        {/* ============================================ */}
        {/* HVAC SYSTEM */}
        {/* ============================================ */}
        <AccordionItem value="hvac" className="border rounded-lg">
          <AccordionTrigger className="px-4 hover:no-underline">
            <div className="flex items-center gap-3">
              <Wind className="h-5 w-5 text-primary" />
              <span className="font-semibold">HVAC System</span>
              <Badge variant="outline" className="ml-2">Required</Badge>
            </div>
          </AccordionTrigger>
          <AccordionContent className="px-4 pb-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              <div className="space-y-2">
                <Label>System Type</Label>
                <Controller
                  name="hvac_system_type"
                  control={form.control}
                  render={({ field }) => (
                    <Select onValueChange={field.onChange} value={field.value}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="vav">Variable Air Volume (VAV)</SelectItem>
                        <SelectItem value="cav">Constant Air Volume (CAV)</SelectItem>
                        <SelectItem value="doas">Dedicated Outdoor Air (DOAS)</SelectItem>
                        <SelectItem value="vrf">Variable Refrigerant Flow (VRF)</SelectItem>
                        <SelectItem value="split">Split System</SelectItem>
                        <SelectItem value="ptac">PTAC/PTHP</SelectItem>
                        <SelectItem value="radiant">Radiant Floor/Ceiling</SelectItem>
                        <SelectItem value="chilled_beam">Chilled Beam</SelectItem>
                        <SelectItem value="other">Other</SelectItem>
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>

              <div className="space-y-2">
                <Label>Diffuser Type</Label>
                <Controller
                  name="diffuser_type"
                  control={form.control}
                  render={({ field }) => (
                    <Select onValueChange={field.onChange} value={field.value}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="ceiling_4way">Ceiling - 4-Way</SelectItem>
                        <SelectItem value="ceiling_2way">Ceiling - 2-Way</SelectItem>
                        <SelectItem value="ceiling_1way">Ceiling - Linear</SelectItem>
                        <SelectItem value="ceiling_round">Ceiling - Round</SelectItem>
                        <SelectItem value="ceiling_perforated">Ceiling - Perforated</SelectItem>
                        <SelectItem value="sidewall">Sidewall Register</SelectItem>
                        <SelectItem value="floor">Floor Diffuser</SelectItem>
                        <SelectItem value="displacement">Displacement</SelectItem>
                        <SelectItem value="slot">Slot Diffuser</SelectItem>
                        <SelectItem value="jet">Jet Nozzle</SelectItem>
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="vent_count">Number of Supply Diffusers *</Label>
                <Input
                  id="vent_count"
                  type="number"
                  min="1"
                  {...form.register('vent_count', { valueAsNumber: true })}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="return_count">Number of Return Grilles</Label>
                <Input
                  id="return_count"
                  type="number"
                  min="0"
                  {...form.register('return_count', { valueAsNumber: true })}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="supply_airflow">Total Supply Airflow ({airflowUnit}) *</Label>
                <Input
                  id="supply_airflow"
                  type="number"
                  step="1"
                  {...form.register('supply_airflow', { valueAsNumber: true })}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="supply_temperature">Supply Air Temperature ({tempUnit}) *</Label>
                <Input
                  id="supply_temperature"
                  type="number"
                  step="0.1"
                  {...form.register('supply_temperature', { valueAsNumber: true })}
                />
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>

        {/* ============================================ */}
        {/* THERMAL LOADS */}
        {/* ============================================ */}
        <AccordionItem value="loads" className="border rounded-lg">
          <AccordionTrigger className="px-4 hover:no-underline">
            <div className="flex items-center gap-3">
              <Users className="h-5 w-5 text-primary" />
              <span className="font-semibold">Thermal Loads</span>
              <Badge variant="secondary" className="ml-2">Recommended</Badge>
            </div>
          </AccordionTrigger>
          <AccordionContent className="px-4 pb-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <div className="space-y-2">
                <Label htmlFor="occupancy">Design Occupancy (people)</Label>
                <Input
                  id="occupancy"
                  type="number"
                  min="0"
                  {...form.register('occupancy', { valueAsNumber: true })}
                />
              </div>

              <div className="space-y-2">
                <Label>Activity Level</Label>
                <Controller
                  name="occupant_activity"
                  control={form.control}
                  render={({ field }) => (
                    <Select onValueChange={field.onChange} value={field.value}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="sleeping">Sleeping (40 W/person)</SelectItem>
                        <SelectItem value="seated_quiet">Seated, Quiet (60 W)</SelectItem>
                        <SelectItem value="office">Office Work (75 W)</SelectItem>
                        <SelectItem value="standing_light">Standing, Light (90 W)</SelectItem>
                        <SelectItem value="walking">Walking (110 W)</SelectItem>
                        <SelectItem value="light_machine">Light Machine (140 W)</SelectItem>
                        <SelectItem value="heavy_work">Heavy Work (235 W)</SelectItem>
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="lighting_load">Lighting Load (W)</Label>
                <Input
                  id="lighting_load"
                  type="number"
                  min="0"
                  {...form.register('lighting_load', { valueAsNumber: true })}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="equipment_load">Equipment Load (W)</Label>
                <Input
                  id="equipment_load"
                  type="number"
                  min="0"
                  {...form.register('equipment_load', { valueAsNumber: true })}
                />
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>

        {/* ============================================ */}
        {/* WALL THERMAL SETTINGS */}
        {/* ============================================ */}
        <AccordionItem value="walls" className="border rounded-lg">
          <AccordionTrigger className="px-4 hover:no-underline">
            <div className="flex items-center gap-3">
              <Thermometer className="h-5 w-5 text-primary" />
              <span className="font-semibold">Wall & Envelope</span>
              <Badge variant="secondary" className="ml-2">Recommended</Badge>
            </div>
          </AccordionTrigger>
          <AccordionContent className="px-4 pb-4">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label>Wall Thermal Model</Label>
                <Controller
                  name="wall_boundary_type"
                  control={form.control}
                  render={({ field }) => (
                    <Select onValueChange={field.onChange} value={field.value}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="adiabatic">
                          Adiabatic (No Heat Transfer - Airflow Only)
                        </SelectItem>
                        <SelectItem value="fixed_temp">
                          Fixed Surface Temperature
                        </SelectItem>
                        <SelectItem value="u_value">
                          U-Value / Heat Flux (Most Accurate)
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  )}
                />
                <p className="text-xs text-muted-foreground">
                  Adiabatic is simplest - use for pure airflow analysis
                </p>
              </div>

              {form.watch('wall_boundary_type') === 'fixed_temp' && (
                <div className="space-y-2">
                  <Label htmlFor="wall_temperature">
                    Wall Surface Temperature ({tempUnit})
                  </Label>
                  <Input
                    id="wall_temperature"
                    type="number"
                    step="0.1"
                    {...form.register('wall_temperature', { valueAsNumber: true })}
                  />
                </div>
              )}

              <div className="space-y-2">
                <Label htmlFor="indoor_setpoint_cooling">
                  Cooling Setpoint ({tempUnit})
                </Label>
                <Input
                  id="indoor_setpoint_cooling"
                  type="number"
                  step="0.5"
                  {...form.register('indoor_setpoint_cooling', { valueAsNumber: true })}
                />
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>

        {/* ============================================ */}
        {/* SOLVER SETTINGS */}
        {/* ============================================ */}
        <AccordionItem value="solver" className="border rounded-lg">
          <AccordionTrigger className="px-4 hover:no-underline">
            <div className="flex items-center gap-3">
              <Settings2 className="h-5 w-5 text-primary" />
              <span className="font-semibold">Simulation Settings</span>
              <Badge variant="outline" className="ml-2">Optional</Badge>
            </div>
          </AccordionTrigger>
          <AccordionContent className="px-4 pb-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              <div className="space-y-2">
                <Label>Turbulence Model</Label>
                <Controller
                  name="turbulence_model"
                  control={form.control}
                  render={({ field }) => (
                    <Select onValueChange={field.onChange} value={field.value}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="k_epsilon">k-ε Standard</SelectItem>
                        <SelectItem value="k_epsilon_rng">k-ε RNG</SelectItem>
                        <SelectItem value="k_omega">k-ω</SelectItem>
                        <SelectItem value="k_omega_sst">k-ω SST</SelectItem>
                        <SelectItem value="spalart_allmaras">Spalart-Allmaras</SelectItem>
                        <SelectItem value="les">LES (Large Eddy)</SelectItem>
                        <SelectItem value="laminar">Laminar</SelectItem>
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>

              <div className="space-y-2">
                <Label>Grid Resolution</Label>
                <Controller
                  name="grid_resolution"
                  control={form.control}
                  render={({ field }) => (
                    <Select onValueChange={field.onChange} value={field.value}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="coarse">Coarse (Fast, ~100k cells)</SelectItem>
                        <SelectItem value="medium">Medium (~500k cells)</SelectItem>
                        <SelectItem value="fine">Fine (~2M cells)</SelectItem>
                        <SelectItem value="very_fine">Very Fine (~10M cells)</SelectItem>
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>

              <div className="flex items-center space-x-2 pt-6">
                <Controller
                  name="steady_state"
                  control={form.control}
                  render={({ field }) => (
                    <Checkbox
                      id="steady_state"
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  )}
                />
                <Label htmlFor="steady_state" className="cursor-pointer">
                  Steady-State Simulation
                </Label>
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>

        {/* ============================================ */}
        {/* COMPLIANCE */}
        {/* ============================================ */}
        <AccordionItem value="compliance" className="border rounded-lg">
          <AccordionTrigger className="px-4 hover:no-underline">
            <div className="flex items-center gap-3">
              <FileCheck className="h-5 w-5 text-primary" />
              <span className="font-semibold">Compliance & Standards</span>
              <Badge variant="outline" className="ml-2">Optional</Badge>
            </div>
          </AccordionTrigger>
          <AccordionContent className="px-4 pb-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              <div className="space-y-2">
                <Label>Ventilation Standard</Label>
                <Controller
                  name="ventilation_standard"
                  control={form.control}
                  render={({ field }) => (
                    <Select onValueChange={field.onChange} value={field.value}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="ashrae_62_1">ASHRAE 62.1</SelectItem>
                        <SelectItem value="ashrae_62_2">ASHRAE 62.2 (Residential)</SelectItem>
                        <SelectItem value="en_16798">EN 16798 (Europe)</SelectItem>
                        <SelectItem value="gb_50736">GB 50736 (China)</SelectItem>
                        <SelectItem value="custom">Custom</SelectItem>
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>

              <div className="space-y-2">
                <Label>Space Type (ASHRAE)</Label>
                <Controller
                  name="space_type_ashrae"
                  control={form.control}
                  render={({ field }) => (
                    <Select onValueChange={field.onChange} value={field.value}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="office">Office Space</SelectItem>
                        <SelectItem value="conference">Conference Room</SelectItem>
                        <SelectItem value="classroom">Classroom</SelectItem>
                        <SelectItem value="retail">Retail Sales</SelectItem>
                        <SelectItem value="restaurant">Restaurant Dining</SelectItem>
                        <SelectItem value="hotel_lobby">Hotel Lobby</SelectItem>
                        <SelectItem value="hotel_room">Hotel Guest Room</SelectItem>
                        <SelectItem value="hospital_patient">Hospital Patient Room</SelectItem>
                        <SelectItem value="laboratory">Laboratory</SelectItem>
                        <SelectItem value="gymnasium">Gymnasium</SelectItem>
                        <SelectItem value="warehouse">Warehouse</SelectItem>
                        <SelectItem value="data_center">Data Center</SelectItem>
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="adpi_target">ADPI Target (%)</Label>
                <Input
                  id="adpi_target"
                  type="number"
                  min="0"
                  max="100"
                  {...form.register('adpi_target', { valueAsNumber: true })}
                />
                <p className="text-xs text-muted-foreground">
                  Air Distribution Performance Index (ASHRAE 55)
                </p>
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>

      {/* ============================================ */}
      {/* FORM ACTIONS */}
      {/* ============================================ */}
      <div className="flex justify-end gap-3 mt-6 pt-4 border-t">
        {onCancel && (
          <Button type="button" variant="outline" onClick={onCancel}>
            Cancel
          </Button>
        )}
        <Button type="submit" disabled={isLoading}>
          {isLoading ? (
            <>
              <span className="animate-spin mr-2">⏳</span>
              Processing...
            </>
          ) : (
            <>
              Continue
              <ChevronRight className="ml-2 h-4 w-4" />
            </>
          )}
        </Button>
      </div>
    </form>
  );
}

export default IntakeForm;

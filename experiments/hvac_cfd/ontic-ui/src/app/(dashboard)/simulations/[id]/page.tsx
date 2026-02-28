/**
 * Simulation Detail Page
 * 
 * Real-time monitoring, control, and analysis of a single simulation.
 */

'use client';

import { useMemo } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  Play,
  Pause,
  Square,
  Download,
  ArrowLeft,
  Clock,
  Cpu,
  Box,
  Activity,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Settings2,
  ChevronRight,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Skeleton } from '@/components/ui/skeleton';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Sidebar,
  Header,
  DashboardShell,
  PageHeader,
  SectionCard,
  EmptyState,
} from '@/components/layout';
import { ResidualChart } from '@/components/cfd/ResidualChart';
import {
  useSimulation,
  useResiduals,
  useStartSimulation,
  usePauseSimulation,
  useStopSimulation,
  useDeleteSimulation,
  useSimulationFields,
  useSimulationWebSocket,
  getExportUrl,
} from '@/hooks';
import { useToast } from '@/components/ui/use-toast';

// ============================================
// STATUS BADGE
// ============================================

function StatusBadge({ status }: { status: string }) {
  const defaultConfig = { variant: 'secondary' as const, label: 'Pending', icon: <Clock className="h-3 w-3" /> };
  const config: Record<string, { variant: 'default' | 'secondary' | 'destructive' | 'outline'; label: string; icon: React.ReactNode }> = {
    pending: defaultConfig,
    running: { variant: 'default', label: 'Running', icon: <Loader2 className="h-3 w-3 animate-spin" /> },
    paused: { variant: 'outline', label: 'Paused', icon: <Pause className="h-3 w-3" /> },
    completed: { variant: 'default', label: 'Completed', icon: <CheckCircle2 className="h-3 w-3" /> },
    failed: { variant: 'destructive', label: 'Failed', icon: <AlertCircle className="h-3 w-3" /> },
  };

  const c = config[status] ?? defaultConfig;

  return (
    <Badge variant={c.variant} className="gap-1">
      {c.icon}
      {c.label}
    </Badge>
  );
}

// ============================================
// SIMULATION CONTROLS
// ============================================

interface SimulationControlsProps {
  simulationId: string;
  status: string;
}

function SimulationControls({ simulationId, status }: SimulationControlsProps) {
  const { toast } = useToast();
  const startSimulation = useStartSimulation();
  const pauseSimulation = usePauseSimulation();
  const stopSimulation = useStopSimulation();

  const handleStart = async () => {
    try {
      await startSimulation.mutateAsync(simulationId);
      toast({ title: 'Simulation started' });
    } catch (err) {
      toast({ title: 'Failed to start', description: (err as Error).message, variant: 'destructive' });
    }
  };

  const handlePause = async () => {
    try {
      await pauseSimulation.mutateAsync(simulationId);
      toast({ title: 'Simulation paused' });
    } catch (err) {
      toast({ title: 'Failed to pause', description: (err as Error).message, variant: 'destructive' });
    }
  };

  const handleStop = async () => {
    try {
      await stopSimulation.mutateAsync(simulationId);
      toast({ title: 'Simulation stopped' });
    } catch (err) {
      toast({ title: 'Failed to stop', description: (err as Error).message, variant: 'destructive' });
    }
  };

  const isLoading = startSimulation.isPending || pauseSimulation.isPending || stopSimulation.isPending;

  return (
    <div className="flex gap-2">
      {(status === 'pending' || status === 'paused') && (
        <Button onClick={handleStart} disabled={isLoading}>
          <Play className="h-4 w-4 mr-2" />
          {status === 'paused' ? 'Resume' : 'Start'}
        </Button>
      )}
      {status === 'running' && (
        <>
          <Button onClick={handlePause} variant="outline" disabled={isLoading}>
            <Pause className="h-4 w-4 mr-2" />
            Pause
          </Button>
          <Button onClick={handleStop} variant="destructive" disabled={isLoading}>
            <Square className="h-4 w-4 mr-2" />
            Stop
          </Button>
        </>
      )}
      {(status === 'completed' || status === 'failed') && (
        <Button asChild variant="outline">
          <a href={getExportUrl(simulationId)} download>
            <Download className="h-4 w-4 mr-2" />
            Export Results
          </a>
        </Button>
      )}
    </div>
  );
}

// ============================================
// SETTINGS PANEL
// ============================================

function SettingsPanel({ simulation }: { simulation: NonNullable<ReturnType<typeof useSimulation>['data']> }) {
  const settings = simulation.settings;
  if (!settings) return null;

  const items = [
    { label: 'Solver Type', value: settings.solverType },
    { label: 'Turbulence Model', value: settings.turbulenceModel },
    { label: 'Max Iterations', value: settings.maxIterations.toLocaleString() },
    { label: 'Convergence Tolerance', value: settings.convergenceTolerance.toExponential(2) },
    { label: 'CFL Number', value: settings.cflNumber },
    { label: 'GPU Acceleration', value: settings.gpuAcceleration ? 'Enabled' : 'Disabled' },
    { label: 'Precision', value: settings.precision },
  ];

  return (
    <div className="grid gap-3">
      {items.map((item) => (
        <div key={item.label} className="flex justify-between text-sm">
          <span className="text-muted-foreground">{item.label}</span>
          <span className="font-medium">{item.value}</span>
        </div>
      ))}
    </div>
  );
}

// ============================================
// MAIN PAGE
// ============================================

export default function SimulationDetailPage({
  params,
}: {
  params: { id: string };
}) {
  const { id } = params;
  const router = useRouter();
  const { toast } = useToast();
  
  const { data: simulation, isLoading, error } = useSimulation(id);
  const { data: residuals } = useResiduals(id);
  const { data: fields } = useSimulationFields(id);
  const { isConnected, liveResiduals } = useSimulationWebSocket(
    simulation?.status === 'running' ? id : null
  );
  const deleteSimulation = useDeleteSimulation();

  // Use live residuals when running, otherwise use fetched residuals
  const displayResiduals = simulation?.status === 'running' && liveResiduals.length > 0
    ? liveResiduals
    : residuals;

  const progress = useMemo(() => {
    if (!simulation) return 0;
    const current = simulation.currentIteration ?? 0;
    const max = simulation.maxIterations ?? 1;
    return Math.round((current / max) * 100);
  }, [simulation]);

  const handleDelete = async () => {
    if (!confirm('Are you sure you want to delete this simulation?')) return;
    try {
      await deleteSimulation.mutateAsync(id);
      toast({ title: 'Simulation deleted' });
      router.push('/simulations');
    } catch (err) {
      toast({ title: 'Delete failed', description: (err as Error).message, variant: 'destructive' });
    }
  };

  if (error) {
    return (
      <div className="flex h-screen overflow-hidden">
        <Sidebar />
        <div className="flex-1 flex flex-col overflow-hidden">
          <Header />
          <DashboardShell>
            <SectionCard>
              <EmptyState
                icon={<AlertCircle className="h-6 w-6" />}
                title="Simulation not found"
                description={error.message}
                action={
                  <Button asChild>
                    <Link href="/simulations">Back to Simulations</Link>
                  </Button>
                }
              />
            </SectionCard>
          </DashboardShell>
        </div>
      </div>
    );
  }

  if (isLoading || !simulation) {
    return (
      <div className="flex h-screen overflow-hidden">
        <Sidebar />
        <div className="flex-1 flex flex-col overflow-hidden">
          <Header />
          <DashboardShell>
            <div className="space-y-6">
              <Skeleton className="h-20" />
              <div className="grid gap-6 lg:grid-cols-3">
                <Skeleton className="h-64 lg:col-span-2" />
                <Skeleton className="h-64" />
              </div>
            </div>
          </DashboardShell>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <DashboardShell>
          <div className="space-y-6">
            {/* Header */}
            <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
              <div>
                <div className="flex items-center gap-2 text-sm text-muted-foreground mb-2">
                  <Link href="/" className="hover:text-foreground">Dashboard</Link>
                  <ChevronRight className="h-4 w-4" />
                  <Link href="/simulations" className="hover:text-foreground">Simulations</Link>
                  <ChevronRight className="h-4 w-4" />
                  <span>{simulation.name}</span>
                </div>
                <div className="flex items-center gap-3">
                  <h1 className="text-2xl font-bold">{simulation.name}</h1>
                  <StatusBadge status={simulation.status} />
                </div>
                {simulation.description && (
                  <p className="text-muted-foreground mt-1">{simulation.description}</p>
                )}
              </div>
              <SimulationControls simulationId={id} status={simulation.status} />
            </div>

            {/* Progress Bar (for running simulations) */}
            {simulation.status === 'running' && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Progress</span>
                  <span>{simulation.currentIteration?.toLocaleString()} / {simulation.maxIterations?.toLocaleString()} iterations</span>
                </div>
                <Progress value={progress} className="h-3" />
              </div>
            )}

            {/* Main Content */}
            <Tabs defaultValue="residuals" className="space-y-6">
              <TabsList>
                <TabsTrigger value="residuals">Residuals</TabsTrigger>
                <TabsTrigger value="settings">Settings</TabsTrigger>
                <TabsTrigger value="fields">Fields</TabsTrigger>
              </TabsList>

              <TabsContent value="residuals" className="space-y-6">
                <div className="grid gap-6 lg:grid-cols-3">
                  {/* Residual Chart */}
                  <SectionCard 
                    title={
                      <div className="flex items-center gap-2">
                        Convergence History
                        {isConnected && simulation?.status === 'running' && (
                          <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" title="Live updates" />
                        )}
                      </div>
                    } 
                    className="lg:col-span-2"
                  >
                    {displayResiduals && displayResiduals.length > 0 ? (
                      <div className="h-80">
                        <ResidualChart data={displayResiduals} />
                      </div>
                    ) : (
                      <EmptyState
                        icon={<Activity className="h-6 w-6" />}
                        title="No residuals yet"
                        description="Start the simulation to see convergence data"
                      />
                    )}
                  </SectionCard>

                  {/* Info Panel */}
                  <SectionCard title="Details">
                    <div className="space-y-4">
                      <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                        <Box className="h-5 w-5 text-purple-500" />
                        <div className="flex-1">
                          <p className="text-xs text-muted-foreground">Mesh</p>
                          <p className="font-medium text-sm">{simulation.meshName ?? 'Unknown'}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                        <Cpu className="h-5 w-5 text-blue-500" />
                        <div className="flex-1">
                          <p className="text-xs text-muted-foreground">Solver</p>
                          <p className="font-medium text-sm">{simulation.settings?.solverType ?? 'Steady'}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                        <Clock className="h-5 w-5 text-amber-500" />
                        <div className="flex-1">
                          <p className="text-xs text-muted-foreground">Created</p>
                          <p className="font-medium text-sm">
                            {new Date(simulation.createdAt).toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                      {simulation.performance && (
                        <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                          <Activity className="h-5 w-5 text-emerald-500" />
                          <div className="flex-1">
                            <p className="text-xs text-muted-foreground">Throughput</p>
                            <p className="font-medium text-sm">
                              {simulation.performance.throughput.toFixed(2)} Mcells/s
                            </p>
                          </div>
                        </div>
                      )}
                    </div>
                  </SectionCard>
                </div>
              </TabsContent>

              <TabsContent value="settings">
                <SectionCard title="Solver Configuration">
                  <SettingsPanel simulation={simulation} />
                </SectionCard>
              </TabsContent>

              <TabsContent value="fields">
                <SectionCard title="Available Fields">
                  {fields && fields.length > 0 ? (
                    <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
                      {fields.map((field) => (
                        <div key={field.name} className="p-4 rounded-lg border">
                          <div className="flex items-center justify-between mb-2">
                            <span className="font-mono font-medium">{field.name}</span>
                            <Badge variant="outline">{field.unit}</Badge>
                          </div>
                          <p className="text-sm text-muted-foreground">{field.description}</p>
                          {field.components && (
                            <div className="flex gap-1 mt-2">
                              {field.components.map((c) => (
                                <Badge key={c} variant="secondary" className="text-xs">{c}</Badge>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <EmptyState
                      icon={<Settings2 className="h-6 w-6" />}
                      title="No fields available"
                      description="Complete the simulation to see field data"
                    />
                  )}
                </SectionCard>
              </TabsContent>
            </Tabs>

            {/* Danger Zone */}
            <SectionCard title="Danger Zone" className="border-destructive/50">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Delete Simulation</p>
                  <p className="text-sm text-muted-foreground">
                    Permanently remove this simulation and all its data
                  </p>
                </div>
                <Button
                  variant="destructive"
                  onClick={handleDelete}
                  disabled={deleteSimulation.isPending}
                >
                  Delete
                </Button>
              </div>
            </SectionCard>
          </div>
        </DashboardShell>
      </div>
    </div>
  );
}

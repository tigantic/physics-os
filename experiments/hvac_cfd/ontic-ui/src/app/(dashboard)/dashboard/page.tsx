/**
 * Dashboard Route
 * 
 * Provides the overview dashboard at /dashboard URL.
 */

'use client';

import { Suspense } from 'react';
import Link from 'next/link';
import {
  Play,
  Box,
  Cpu,
  BarChart3,
  Zap,
  Plus,
  ArrowRight,
  Activity,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Sidebar,
  Header,
  DashboardShell,
  PageHeader,
  SectionCard,
  StatCard,
  EmptyState,
} from '@/components/layout';
import { SimulationCard } from '@/components/cfd';
import { ResidualChartMini } from '@/components/cfd/ResidualChart';
import { useSimulations, useSystemStatus, useGPUs, useMeshes, useActivities } from '@/hooks';
import { useSimulationStore } from '@/stores';

function DashboardStats() {
  const { data: simulations, isLoading } = useSimulations();
  const { data: meshes } = useMeshes();
  const { data: systemStatus } = useSystemStatus();
  const { data: gpus } = useGPUs();

  if (isLoading) {
    return (
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-32 rounded-xl" />
        ))}
      </div>
    );
  }

  const runningCount = simulations?.filter((s) => s.status === 'running').length ?? 0;
  const completedCount = simulations?.filter((s) => s.status === 'completed').length ?? 0;
  const meshCount = meshes?.length ?? 0;
  const gpuUtil = systemStatus?.gpuUtilization ?? 0;

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      <StatCard
        title="Active Simulations"
        value={runningCount}
        description={`${simulations?.length ?? 0} total simulations`}
        icon={<Play className="h-4 w-4" />}
      />
      <StatCard
        title="Completed Today"
        value={completedCount}
        description="Simulations finished"
        icon={<CheckCircle2 className="h-4 w-4" />}
        trend={{ value: 15, isPositive: true }}
      />
      <StatCard
        title="Meshes"
        value={meshCount}
        description="Ready for simulation"
        icon={<Box className="h-4 w-4" />}
      />
      <StatCard
        title="GPU Utilization"
        value={`${gpuUtil.toFixed(0)}%`}
        description={gpus?.[0]?.name ?? 'No GPU detected'}
        icon={<Cpu className="h-4 w-4" />}
      />
    </div>
  );
}

function ActiveSimulations() {
  const { data: simulations, isLoading } = useSimulations();
  const residuals = useSimulationStore((s) => s.residuals);

  const activeSimulations = simulations?.filter(
    (s) => s.status === 'running' || s.status === 'paused'
  ) ?? [];

  if (isLoading) {
    return (
      <SectionCard
        title="Active Simulations"
        actions={
          <Button size="sm" variant="outline" disabled>
            <Plus className="h-4 w-4 mr-1" /> New
          </Button>
        }
      >
        <div className="space-y-4">
          {Array.from({ length: 2 }).map((_, i) => (
            <Skeleton key={i} className="h-24 rounded-lg" />
          ))}
        </div>
      </SectionCard>
    );
  }

  return (
    <SectionCard
      title="Active Simulations"
      description={`${activeSimulations.length} simulation${activeSimulations.length !== 1 ? 's' : ''} in progress`}
      actions={
        <Button size="sm" asChild>
          <Link href="/simulations/new">
            <Plus className="h-4 w-4 mr-1" /> New Simulation
          </Link>
        </Button>
      }
    >
      {activeSimulations.length === 0 ? (
        <EmptyState
          icon={<Play className="h-6 w-6" />}
          title="No active simulations"
          description="Start a new simulation to see real-time progress here."
          action={
            <Button asChild>
              <Link href="/simulations/new">
                <Plus className="h-4 w-4 mr-2" /> Create Simulation
              </Link>
            </Button>
          }
        />
      ) : (
        <div className="space-y-4">
          {activeSimulations.map((sim) => (
            <div key={sim.id} className="flex gap-4">
              <div className="flex-1">
                <SimulationCard simulation={sim} compact />
              </div>
              <div className="w-64 hidden lg:block">
                <ResidualChartMini
                  data={residuals}
                  height={80}
                  className="bg-muted/30 rounded-lg p-2"
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </SectionCard>
  );
}

function GPUStatus() {
  const { data: gpus, isLoading } = useGPUs();

  if (isLoading) {
    return (
      <SectionCard title="GPU Status">
        <Skeleton className="h-24 rounded-lg" />
      </SectionCard>
    );
  }

  if (!gpus || gpus.length === 0) {
    return (
      <SectionCard title="GPU Status">
        <EmptyState
          icon={<Cpu className="h-6 w-6" />}
          title="No GPU detected"
          description="CUDA-capable GPU required for HyperGrid acceleration."
        />
      </SectionCard>
    );
  }

  return (
    <SectionCard title="GPU Status">
      <div className="space-y-4">
        {gpus.map((gpu, index) => (
          <div key={index} className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-emerald-500" />
                <span className="font-medium">{gpu.name}</span>
              </div>
              <span className="text-sm text-muted-foreground">
                {gpu.memoryUsed.toFixed(1)} / {gpu.memoryTotal.toFixed(1)} GB
              </span>
            </div>
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Memory</span>
                <span>{((gpu.memoryUsed / gpu.memoryTotal) * 100).toFixed(0)}%</span>
              </div>
              <Progress
                value={(gpu.memoryUsed / gpu.memoryTotal) * 100}
                className="h-2"
              />
            </div>
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Utilization</span>
                <span>{gpu.utilization.toFixed(0)}%</span>
              </div>
              <Progress value={gpu.utilization} className="h-2" />
            </div>
          </div>
        ))}
      </div>
    </SectionCard>
  );
}

function QuickActions() {
  const actions = [
    { label: 'New Simulation', href: '/simulations/new', icon: Play, description: 'Start a CFD simulation' },
    { label: 'View Meshes', href: '/meshes', icon: Box, description: 'Manage & import meshes' },
    { label: 'View Results', href: '/results', icon: BarChart3, description: 'Analyze completed runs' },
    { label: 'System Status', href: '/status', icon: Zap, description: 'GPU & performance' },
  ];

  return (
    <SectionCard title="Quick Actions">
      <div className="grid gap-3 sm:grid-cols-2">
        {actions.map((action) => (
          <Link
            key={action.href}
            href={action.href}
            className="flex items-center gap-3 p-3 rounded-lg border hover:bg-muted/50 transition-colors group"
          >
            <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
              <action.icon className="h-5 w-5" />
            </div>
            <div className="flex-1">
              <div className="font-medium text-sm">{action.label}</div>
              <div className="text-xs text-muted-foreground">{action.description}</div>
            </div>
            <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:text-foreground transition-colors" />
          </Link>
        ))}
      </div>
    </SectionCard>
  );
}

function RecentActivity() {
  const { data: activities, isLoading } = useActivities(10);

  const getIcon = (type: string) => {
    switch (type) {
      case 'simulation_completed': return <CheckCircle2 className="h-4 w-4 text-emerald-500" />;
      case 'simulation_started': return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'simulation_failed': return <XCircle className="h-4 w-4 text-red-500" />;
      case 'mesh_imported': return <Box className="h-4 w-4 text-purple-500" />;
      default: return <Activity className="h-4 w-4" />;
    }
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);
    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins} minute${diffMins === 1 ? '' : 's'} ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours === 1 ? '' : 's'} ago`;
    return `${diffDays} day${diffDays === 1 ? '' : 's'} ago`;
  };

  if (isLoading) {
    return (
      <SectionCard title="Recent Activity">
        <div className="space-y-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="flex items-start gap-3">
              <Skeleton className="h-4 w-4 rounded-full" />
              <div className="flex-1 space-y-2">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-3 w-20" />
              </div>
            </div>
          ))}
        </div>
      </SectionCard>
    );
  }

  return (
    <SectionCard title="Recent Activity">
      <div className="space-y-4">
        {(activities ?? []).length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-4">No recent activity</p>
        ) : (
          (activities ?? []).map((activity) => (
            <div key={activity.id} className="flex items-start gap-3">
              <div className="mt-0.5">{getIcon(activity.type)}</div>
              <div className="flex-1 min-w-0">
                <p className="text-sm truncate">{activity.title}</p>
                <p className="text-xs text-muted-foreground flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {formatTime(activity.timestamp)}
                </p>
              </div>
            </div>
          ))
        )}
      </div>
    </SectionCard>
  );
}

export default function DashboardPage() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <DashboardShell>
          <div className="space-y-6">
            <PageHeader title="Dashboard" description="Monitor your HyperGrid simulations and system status" />
            <Suspense fallback={<Skeleton className="h-32" />}>
              <DashboardStats />
            </Suspense>
            <div className="grid gap-6 lg:grid-cols-3">
              <div className="lg:col-span-2">
                <Suspense fallback={<Skeleton className="h-64" />}>
                  <ActiveSimulations />
                </Suspense>
              </div>
              <div>
                <Suspense fallback={<Skeleton className="h-64" />}>
                  <GPUStatus />
                </Suspense>
              </div>
            </div>
            <div className="grid gap-6 lg:grid-cols-2">
              <QuickActions />
              <RecentActivity />
            </div>
          </div>
        </DashboardShell>
      </div>
    </div>
  );
}


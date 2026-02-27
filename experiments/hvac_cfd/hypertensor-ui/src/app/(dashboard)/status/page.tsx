/**
 * System Status Page
 * 
 * Displays GPU status, system metrics, and solver health.
 */

'use client';

import { Cpu, HardDrive, Activity, Thermometer, Zap, Server } from 'lucide-react';
import { Progress } from '@/components/ui/progress';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import {
  Sidebar,
  Header,
  DashboardShell,
  PageHeader,
  SectionCard,
  StatCard,
} from '@/components/layout';
import { useSystemStatus, useGPUs } from '@/hooks';

// ============================================
// GPU CARD
// ============================================

interface GPUCardProps {
  gpu: {
    index: number;
    name: string;
    memoryTotal: number;
    memoryUsed: number;
    utilization: number;
    temperature?: number;
  };
}

function GPUCard({ gpu }: GPUCardProps) {
  const memoryPercent = (gpu.memoryUsed / gpu.memoryTotal) * 100;
  const formatMemory = (bytes: number) => `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`;

  return (
    <div className="rounded-xl border bg-card p-4">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="font-medium">{gpu.name}</h3>
          <p className="text-xs text-muted-foreground">GPU {gpu.index}</p>
        </div>
        <Badge variant={gpu.utilization > 80 ? 'default' : 'secondary'}>
          {gpu.utilization.toFixed(0)}% Active
        </Badge>
      </div>

      <div className="space-y-4">
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-muted-foreground">Memory</span>
            <span>{formatMemory(gpu.memoryUsed)} / {formatMemory(gpu.memoryTotal)}</span>
          </div>
          <Progress value={memoryPercent} />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="flex items-center gap-2">
            <Zap className="h-4 w-4 text-yellow-500" />
            <span className="text-sm">{gpu.utilization.toFixed(0)}% Compute</span>
          </div>
          <div className="flex items-center gap-2">
            <Thermometer className="h-4 w-4 text-orange-500" />
            <span className="text-sm">{gpu.temperature ?? 'N/A'}°C</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================
// MAIN PAGE
// ============================================

export default function StatusPage() {
  const { data: systemStatus, isLoading: statusLoading } = useSystemStatus();
  const { data: gpus, isLoading: gpusLoading } = useGPUs();

  const isLoading = statusLoading || gpusLoading;

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <DashboardShell>
          <div className="space-y-6">
            <PageHeader
              title="System Status"
              description="Monitor GPU resources and solver health"
              breadcrumbs={[
                { label: 'Dashboard', href: '/' },
                { label: 'System Status' },
              ]}
            />

            {/* Stats Overview */}
            {isLoading ? (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                {Array.from({ length: 4 }).map((_, i) => (
                  <Skeleton key={i} className="h-32 rounded-xl" />
                ))}
              </div>
            ) : (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <StatCard
                  title="GPU Count"
                  value={gpus?.length ?? 0}
                  description="CUDA-capable devices"
                  icon={<Cpu className="h-4 w-4" />}
                />
                <StatCard
                  title="GPU Utilization"
                  value={`${systemStatus?.gpuUtilization?.toFixed(0) ?? 0}%`}
                  description="Average compute usage"
                  icon={<Activity className="h-4 w-4" />}
                />
                <StatCard
                  title="Memory Used"
                  value={`${systemStatus?.memoryUsedGb?.toFixed(1) ?? 0} GB`}
                  description="Total GPU memory in use"
                  icon={<HardDrive className="h-4 w-4" />}
                />
                <StatCard
                  title="Active Simulations"
                  value={systemStatus?.activeSimulations ?? 0}
                  description="Currently running"
                  icon={<Server className="h-4 w-4" />}
                />
              </div>
            )}

            {/* GPU Details */}
            <SectionCard title="GPU Devices" description="Individual GPU status and metrics">
              {gpusLoading ? (
                <div className="grid gap-4 md:grid-cols-2">
                  {Array.from({ length: 2 }).map((_, i) => (
                    <Skeleton key={i} className="h-40 rounded-xl" />
                  ))}
                </div>
              ) : gpus && gpus.length > 0 ? (
                <div className="grid gap-4 md:grid-cols-2">
                  {gpus.map((gpu) => (
                    <GPUCard key={gpu.index} gpu={gpu} />
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <Cpu className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No CUDA-capable GPUs detected</p>
                  <p className="text-sm">HyperGrid requires NVIDIA GPU with CUDA support</p>
                </div>
              )}
            </SectionCard>
          </div>
        </DashboardShell>
      </div>
    </div>
  );
}

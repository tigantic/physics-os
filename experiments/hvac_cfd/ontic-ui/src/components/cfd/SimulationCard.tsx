/**
 * SimulationCard - Active Simulation Status Display
 * 
 * Compact card showing simulation status, progress, and performance metrics.
 * Used in dashboard for quick overview of running simulations.
 */

'use client';

import { useMemo } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Play, 
  Pause, 
  Square, 
  CheckCircle, 
  XCircle, 
  Clock,
  Cpu,
  Gauge,
  HardDrive
} from 'lucide-react';
import type { Simulation, SimulationSummary, SimulationStatus } from '@/types';

// ============================================
// PROPS INTERFACE
// ============================================

type SimulationLike = Simulation | (SimulationSummary & {
  iteration?: number;
  max_iterations?: number;
  current_time?: number;
  end_time?: number;
});

interface SimulationCardProps {
  simulation: SimulationLike;
  compact?: boolean;
  className?: string;
}

// ============================================
// STATUS CONFIGURATION
// ============================================

const STATUS_CONFIG: Record<SimulationStatus, {
  icon: typeof Play;
  color: string;
  badgeVariant: 'default' | 'secondary' | 'destructive' | 'outline';
}> = {
  pending: {
    icon: Clock,
    color: 'text-slate-500',
    badgeVariant: 'secondary',
  },
  running: {
    icon: Play,
    color: 'text-green-500',
    badgeVariant: 'default',
  },
  paused: {
    icon: Pause,
    color: 'text-amber-500',
    badgeVariant: 'outline',
  },
  completed: {
    icon: CheckCircle,
    color: 'text-blue-500',
    badgeVariant: 'secondary',
  },
  failed: {
    icon: XCircle,
    color: 'text-red-500',
    badgeVariant: 'destructive',
  },
};

// ============================================
// MAIN COMPONENT
// ============================================

export function SimulationCard({ 
  simulation, 
  compact = false,
  className = '' 
}: SimulationCardProps) {
  const statusConfig = STATUS_CONFIG[simulation.status];
  const StatusIcon = statusConfig.icon;
  
  // Handle both Simulation and SimulationSummary with safe defaults
  const iteration = ('iteration' in simulation ? simulation.iteration : simulation.currentIteration) ?? 0;
  const maxIterations = ('max_iterations' in simulation ? simulation.max_iterations : simulation.maxIterations) ?? 1;
  const currentTime = ('current_time' in simulation ? simulation.current_time : 0) ?? 0;
  const endTime = ('end_time' in simulation ? simulation.end_time : 1) ?? 1;

  const progress = useMemo(() => {
    return Math.min(100, ((iteration ?? 0) / (maxIterations ?? 1)) * 100);
  }, [iteration, maxIterations]);

  const timeProgress = useMemo(() => {
    return Math.min(100, ((currentTime ?? 0) / (endTime ?? 1)) * 100);
  }, [currentTime, endTime]);

  if (compact) {
    return (
      <Card className={`${className}`}>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <StatusIcon className={`h-5 w-5 ${statusConfig.color}`} />
              <div>
                <div className="font-medium">{simulation.name}</div>
                <div className="text-xs text-muted-foreground">
                  {(iteration ?? 0).toLocaleString()} / {(maxIterations ?? 1).toLocaleString()}
                </div>
              </div>
            </div>
            <Badge variant={statusConfig.badgeVariant}>
              {simulation.status}
            </Badge>
          </div>
          <Progress value={progress} className="mt-3 h-1.5" />
        </CardContent>
      </Card>
    );
  }
  
  // Access performance from Simulation type only
  const performance = 'performance' in simulation ? simulation.performance : undefined;
  // Access error from Simulation type only
  const error = 'error' in simulation ? simulation.error : undefined;

  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <StatusIcon className={`h-5 w-5 ${statusConfig.color}`} />
            {simulation.name}
          </CardTitle>
          <Badge variant={statusConfig.badgeVariant} className="capitalize">
            {simulation.status}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Progress */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Progress</span>
            <span className="font-mono">
              {(iteration ?? 0).toLocaleString()} / {(maxIterations ?? 1).toLocaleString()}
            </span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>

        {/* Time Progress */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Simulation Time</span>
            <span className="font-mono">
              {(currentTime ?? 0).toFixed(4)}s / {endTime ?? 1}s
            </span>
          </div>
          <Progress value={timeProgress} className="h-2" />
        </div>

        {/* Performance Metrics */}
        {performance && (
          <div className="grid grid-cols-2 gap-4 pt-2 border-t">
            <MetricItem
              icon={Gauge}
              label="Throughput"
              value={`${performance.throughput.toFixed(1)} Mcells/s`}
            />
            <MetricItem
              icon={Cpu}
              label="GPU Util"
              value={`${performance.gpuUtilization.toFixed(0)}%`}
            />
            <MetricItem
              icon={HardDrive}
              label="VRAM"
              value={`${performance.vramUsedGb.toFixed(1)} / ${performance.vramTotalGb.toFixed(0)} GB`}
            />
            <MetricItem
              icon={Clock}
              label="Wall Time"
              value={formatDuration(performance.wallTimeSeconds)}
            />
          </div>
        )}

        {/* Error Message */}
        {simulation.status === 'failed' && error && (
          <div className="p-3 rounded-md bg-destructive/10 text-destructive text-sm">
            {error}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ============================================
// HELPER COMPONENTS
// ============================================

interface MetricItemProps {
  icon: typeof Gauge;
  label: string;
  value: string;
}

function MetricItem({ icon: Icon, label, value }: MetricItemProps) {
  return (
    <div className="flex items-center gap-2">
      <Icon className="h-4 w-4 text-muted-foreground" />
      <div>
        <div className="text-xs text-muted-foreground">{label}</div>
        <div className="text-sm font-medium font-mono">{value}</div>
      </div>
    </div>
  );
}

// ============================================
// UTILITIES
// ============================================

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  if (seconds < 3600) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
  }
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${mins}m`;
}

/**
 * RunControls - Simulation Control Panel
 * 
 * Start/Pause/Resume/Stop controls with real-time status.
 */

'use client';

import { useState } from 'react';
import {
  Play,
  Pause,
  Square,
  RotateCcw,
  AlertTriangle,
  CheckCircle,
  Loader2,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { useSimulationStore } from '@/stores';
import {
  useStartSimulation,
  usePauseSimulation,
  useStopSimulation,
} from '@/hooks';
import type { SimulationStatus } from '@/types';

// ============================================
// PROPS INTERFACE
// ============================================

interface RunControlsProps {
  simulationId: string;
  status: SimulationStatus;
  className?: string;
  showLabels?: boolean;
}

// ============================================
// MAIN COMPONENT
// ============================================

export function RunControls({
  simulationId,
  status,
  className = '',
  showLabels = false,
}: RunControlsProps) {
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);

  // Mutations
  const startMutation = useStartSimulation();
  const pauseMutation = usePauseSimulation();
  const stopMutation = useStopSimulation();

  // Determine available actions
  const canStart = status === 'pending' || status === 'paused';
  const canPause = status === 'running';
  const canStop = status === 'running' || status === 'paused';
  const canRestart = status === 'completed' || status === 'failed';

  const isLoading =
    startMutation.isPending ||
    pauseMutation.isPending ||
    stopMutation.isPending;

  // Handlers
  const handleStart = () => {
    startMutation.mutate(simulationId);
  };

  const handlePause = () => {
    pauseMutation.mutate(simulationId);
  };

  const handleStop = () => {
    stopMutation.mutate(simulationId);
    setIsConfirmOpen(false);
  };

  const handleRestart = () => {
    startMutation.mutate(simulationId);
  };

  return (
    <TooltipProvider delayDuration={0}>
      <div className={`flex items-center gap-2 ${className}`}>
        {/* Start / Resume Button */}
        {(canStart || canRestart) && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                size={showLabels ? 'default' : 'icon'}
                variant="default"
                onClick={canRestart ? handleRestart : handleStart}
                disabled={isLoading}
                className="bg-emerald-600 hover:bg-emerald-700"
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : canRestart ? (
                  <RotateCcw className="h-4 w-4" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
                {showLabels && (
                  <span className="ml-2">
                    {canRestart ? 'Restart' : status === 'paused' ? 'Resume' : 'Start'}
                  </span>
                )}
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              {canRestart ? 'Restart simulation' : status === 'paused' ? 'Resume simulation' : 'Start simulation'}
            </TooltipContent>
          </Tooltip>
        )}

        {/* Pause Button */}
        {canPause && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                size={showLabels ? 'default' : 'icon'}
                variant="outline"
                onClick={handlePause}
                disabled={isLoading}
              >
                {pauseMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Pause className="h-4 w-4" />
                )}
                {showLabels && <span className="ml-2">Pause</span>}
              </Button>
            </TooltipTrigger>
            <TooltipContent>Pause simulation</TooltipContent>
          </Tooltip>
        )}

        {/* Stop Button */}
        {canStop && (
          <AlertDialog open={isConfirmOpen} onOpenChange={setIsConfirmOpen}>
            <AlertDialogTrigger asChild>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    size={showLabels ? 'default' : 'icon'}
                    variant="destructive"
                    disabled={isLoading}
                  >
                    {stopMutation.isPending ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Square className="h-4 w-4" />
                    )}
                    {showLabels && <span className="ml-2">Stop</span>}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Stop simulation</TooltipContent>
              </Tooltip>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5 text-yellow-500" />
                  Stop Simulation?
                </AlertDialogTitle>
                <AlertDialogDescription>
                  This will terminate the simulation. You can review results up to
                  the current iteration, but the simulation cannot be resumed.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  onClick={handleStop}
                  className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                >
                  Stop Simulation
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        )}

        {/* Status indicator for completed/failed */}
        {status === 'completed' && !showLabels && (
          <div className="flex items-center gap-1 text-emerald-600">
            <CheckCircle className="h-4 w-4" />
          </div>
        )}
        {status === 'failed' && !showLabels && (
          <div className="flex items-center gap-1 text-destructive">
            <AlertTriangle className="h-4 w-4" />
          </div>
        )}
      </div>
    </TooltipProvider>
  );
}

// ============================================
// COMPACT VARIANT
// ============================================

interface RunControlsCompactProps {
  simulationId: string;
  status: SimulationStatus;
  className?: string;
}

export function RunControlsCompact({
  simulationId,
  status,
  className = '',
}: RunControlsCompactProps) {
  const startMutation = useStartSimulation();
  const pauseMutation = usePauseSimulation();

  const isLoading = startMutation.isPending || pauseMutation.isPending;

  if (status === 'running') {
    return (
      <Button
        size="sm"
        variant="ghost"
        onClick={() => pauseMutation.mutate(simulationId)}
        disabled={isLoading}
        className={className}
      >
        {isLoading ? (
          <Loader2 className="h-3 w-3 animate-spin" />
        ) : (
          <Pause className="h-3 w-3" />
        )}
      </Button>
    );
  }

  if (status === 'pending' || status === 'paused') {
    return (
      <Button
        size="sm"
        variant="ghost"
        onClick={() => startMutation.mutate(simulationId)}
        disabled={isLoading}
        className={`text-emerald-600 hover:text-emerald-700 ${className}`}
      >
        {isLoading ? (
          <Loader2 className="h-3 w-3 animate-spin" />
        ) : (
          <Play className="h-3 w-3" />
        )}
      </Button>
    );
  }

  return null;
}

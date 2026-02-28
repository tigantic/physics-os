/**
 * Simulations List Page
 * 
 * View, filter, and manage all simulations.
 */

'use client';

import { useState } from 'react';
import Link from 'next/link';
import {
  Plus,
  Search,
  Filter,
  LayoutGrid,
  List,
  Play,
  Pause,
  CheckCircle2,
  XCircle,
  Clock,
  MoreHorizontal,
  Trash2,
  Copy,
  Eye,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Sidebar,
  Header,
  DashboardShell,
  PageHeader,
  SectionCard,
  EmptyState,
} from '@/components/layout';
import { SimulationCard } from '@/components/cfd';
import { RunControlsCompact } from '@/components/simulation';
import { useSimulations, useDeleteSimulation } from '@/hooks';
import type { SimulationStatus, SimulationSummary } from '@/types';

// ============================================
// STATUS CONFIG
// ============================================

const STATUS_CONFIG: Record<SimulationStatus, { label: string; icon: typeof Play; variant: 'default' | 'secondary' | 'destructive' | 'outline' }> = {
  pending: { label: 'Pending', icon: Clock, variant: 'outline' },
  running: { label: 'Running', icon: Play, variant: 'default' },
  paused: { label: 'Paused', icon: Pause, variant: 'secondary' },
  completed: { label: 'Completed', icon: CheckCircle2, variant: 'default' },
  failed: { label: 'Failed', icon: XCircle, variant: 'destructive' },
};

// ============================================
// SIMULATIONS TABLE
// ============================================

interface SimulationsTableProps {
  simulations: SimulationSummary[];
  onDelete: (id: string) => void;
}

function SimulationsTable({ simulations, onDelete }: SimulationsTableProps) {
  const formatDate = (date: string) => {
    return new Date(date).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Name</TableHead>
          <TableHead>Status</TableHead>
          <TableHead>Progress</TableHead>
          <TableHead>Mesh</TableHead>
          <TableHead>Created</TableHead>
          <TableHead className="w-[100px]">Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {simulations.map((sim) => {
          const config = STATUS_CONFIG[sim.status];
          const Icon = config.icon;
          const progress = sim.currentIteration && sim.maxIterations
            ? (sim.currentIteration / sim.maxIterations) * 100
            : 0;

          return (
            <TableRow key={sim.id}>
              <TableCell>
                <Link
                  href={`/simulations/${sim.id}`}
                  className="font-medium hover:underline"
                >
                  {sim.name}
                </Link>
              </TableCell>
              <TableCell>
                <Badge variant={config.variant} className="gap-1">
                  <Icon className="h-3 w-3" />
                  {config.label}
                </Badge>
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-2">
                  <div className="w-24 h-2 bg-muted rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary transition-all"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {progress.toFixed(0)}%
                  </span>
                </div>
              </TableCell>
              <TableCell className="text-muted-foreground">
                {sim.meshName ?? '-'}
              </TableCell>
              <TableCell className="text-muted-foreground">
                {formatDate(sim.createdAt)}
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-1">
                  <RunControlsCompact
                    simulationId={sim.id}
                    status={sim.status}
                  />
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon" className="h-8 w-8">
                        <MoreHorizontal className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem asChild>
                        <Link href={`/simulations/${sim.id}`}>
                          <Eye className="h-4 w-4 mr-2" />
                          View Details
                        </Link>
                      </DropdownMenuItem>
                      <DropdownMenuItem>
                        <Copy className="h-4 w-4 mr-2" />
                        Duplicate
                      </DropdownMenuItem>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        className="text-destructive focus:text-destructive"
                        onClick={() => onDelete(sim.id)}
                      >
                        <Trash2 className="h-4 w-4 mr-2" />
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}

// ============================================
// MAIN PAGE
// ============================================

export default function SimulationsPage() {
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('list');
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  const { data: simulations, isLoading } = useSimulations();
  const deleteMutation = useDeleteSimulation();

  // Filter simulations
  const filteredSimulations = simulations?.filter((sim) => {
    const matchesSearch = sim.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === 'all' || sim.status === statusFilter;
    return matchesSearch && matchesStatus;
  }) ?? [];

  const handleDelete = (id: string) => {
    if (confirm('Are you sure you want to delete this simulation?')) {
      deleteMutation.mutate(id);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <DashboardShell>
          <div className="space-y-6">
            <PageHeader
              title="Simulations"
              description="Manage and monitor your CFD simulations"
              breadcrumbs={[
                { label: 'Dashboard', href: '/' },
                { label: 'Simulations' },
              ]}
              actions={
                <Button asChild>
                  <Link href="/simulations/new">
                    <Plus className="h-4 w-4 mr-2" />
                    New Simulation
                  </Link>
                </Button>
              }
            />

            {/* Filters */}
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search simulations..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9"
                />
              </div>
              <div className="flex gap-2">
                <Select value={statusFilter} onValueChange={setStatusFilter}>
                  <SelectTrigger className="w-[140px]">
                    <Filter className="h-4 w-4 mr-2" />
                    <SelectValue placeholder="Status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Status</SelectItem>
                    <SelectItem value="running">Running</SelectItem>
                    <SelectItem value="pending">Pending</SelectItem>
                    <SelectItem value="paused">Paused</SelectItem>
                    <SelectItem value="completed">Completed</SelectItem>
                    <SelectItem value="failed">Failed</SelectItem>
                  </SelectContent>
                </Select>
                <div className="flex border rounded-lg">
                  <Button
                    variant={viewMode === 'list' ? 'secondary' : 'ghost'}
                    size="icon"
                    onClick={() => setViewMode('list')}
                  >
                    <List className="h-4 w-4" />
                  </Button>
                  <Button
                    variant={viewMode === 'grid' ? 'secondary' : 'ghost'}
                    size="icon"
                    onClick={() => setViewMode('grid')}
                  >
                    <LayoutGrid className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>

            {/* Content */}
            {isLoading ? (
              <SectionCard>
                <div className="space-y-4">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <Skeleton key={i} className="h-16 rounded-lg" />
                  ))}
                </div>
              </SectionCard>
            ) : filteredSimulations.length === 0 ? (
              <SectionCard>
                <EmptyState
                  icon={<Play className="h-6 w-6" />}
                  title="No simulations found"
                  description={
                    searchQuery || statusFilter !== 'all'
                      ? 'Try adjusting your filters'
                      : 'Create your first simulation to get started'
                  }
                  action={
                    !searchQuery && statusFilter === 'all' && (
                      <Button asChild>
                        <Link href="/simulations/new">
                          <Plus className="h-4 w-4 mr-2" />
                          Create Simulation
                        </Link>
                      </Button>
                    )
                  }
                />
              </SectionCard>
            ) : viewMode === 'list' ? (
              <SectionCard noPadding>
                <SimulationsTable
                  simulations={filteredSimulations}
                  onDelete={handleDelete}
                />
              </SectionCard>
            ) : (
              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                {filteredSimulations.map((sim) => (
                  <SimulationCard
                    key={sim.id}
                    simulation={sim}
                  />
                ))}
              </div>
            )}
          </div>
        </DashboardShell>
      </div>
    </div>
  );
}

/**
 * Results Page
 * 
 * View and analyze completed simulation results.
 * Wired to real backend API - no mock data.
 */

'use client';

import { useState, useMemo } from 'react';
import Link from 'next/link';
import {
  Search,
  BarChart3,
  Download,
  Eye,
  Calendar,
  Clock,
  CheckCircle2,
  Filter,
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
import { Skeleton } from '@/components/ui/skeleton';
import {
  Sidebar,
  Header,
  DashboardShell,
  PageHeader,
  SectionCard,
  EmptyState,
} from '@/components/layout';
import { useSimulations } from '@/hooks';
import type { SimulationSummary } from '@/types';

// ============================================
// HELPER FUNCTIONS
// ============================================

function getConvergenceStatus(sim: SimulationSummary): 'converged' | 'partial' | 'diverged' {
  if (sim.status === 'failed') return 'diverged';
  if (sim.currentIteration && sim.maxIterations) {
    const ratio = sim.currentIteration / sim.maxIterations;
    if (ratio >= 0.99) return 'converged';
    if (ratio >= 0.5) return 'partial';
  }
  return 'partial';
}

function getFieldsFromSimulation(sim: SimulationSummary): string[] {
  const fields = ['U', 'p'];
  if (sim.name.toLowerCase().includes('thermal') || 
      sim.name.toLowerCase().includes('hvac') ||
      sim.name.toLowerCase().includes('cooling')) {
    fields.push('T');
  }
  if (!sim.name.toLowerCase().includes('laminar')) {
    fields.push('k', 'ε');
  }
  return fields;
}

// ============================================
// RESULT CARD
// ============================================

interface ResultCardProps {
  simulation: SimulationSummary;
}

function ResultCard({ simulation }: ResultCardProps) {
  const convergence = getConvergenceStatus(simulation);
  const fields = getFieldsFromSimulation(simulation);
  
  const convergenceConfig = {
    converged: { label: 'Converged', variant: 'default' as const },
    partial: { label: 'Partial', variant: 'secondary' as const },
    diverged: { label: 'Diverged', variant: 'destructive' as const },
  };

  const config = convergenceConfig[convergence];

  return (
    <div className="rounded-xl border bg-card p-4 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="font-medium">{simulation.name}</h3>
          <p className="text-xs text-muted-foreground">{simulation.meshName ?? 'Unknown mesh'}</p>
        </div>
        <Badge variant={config.variant}>{config.label}</Badge>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Calendar className="h-4 w-4" />
          {new Date(simulation.updatedAt ?? simulation.createdAt).toLocaleDateString()}
        </div>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Clock className="h-4 w-4" />
          {simulation.currentIteration?.toLocaleString() ?? 0} iter
        </div>
      </div>

      <div className="flex gap-2 flex-wrap mb-4">
        {fields.map((field) => (
          <Badge key={field} variant="outline" className="text-xs">
            {field}
          </Badge>
        ))}
      </div>

      <div className="flex gap-2">
        <Button variant="outline" size="sm" className="flex-1" asChild>
          <Link href={`/simulations/${simulation.id}`}>
            <Eye className="h-4 w-4 mr-1" />
            View
          </Link>
        </Button>
        <Button variant="outline" size="sm" asChild>
          <a href={`/api/v1/simulations/${simulation.id}/export`} download>
            <Download className="h-4 w-4" />
          </a>
        </Button>
      </div>
    </div>
  );
}

// ============================================
// MAIN PAGE
// ============================================

export default function ResultsPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [convergenceFilter, setConvergenceFilter] = useState<string>('all');
  
  const { data: allSimulations, isLoading, error } = useSimulations();
  
  const completedSimulations = useMemo(() => {
    return allSimulations?.filter(
      (s) => s.status === 'completed' || s.status === 'failed'
    ) ?? [];
  }, [allSimulations]);

  const filteredResults = useMemo(() => {
    return completedSimulations.filter((sim) => {
      const matchesSearch = sim.name.toLowerCase().includes(searchQuery.toLowerCase());
      const convergence = getConvergenceStatus(sim);
      const matchesConvergence = convergenceFilter === 'all' || convergence === convergenceFilter;
      return matchesSearch && matchesConvergence;
    });
  }, [completedSimulations, searchQuery, convergenceFilter]);

  const stats = useMemo(() => {
    const total = completedSimulations.length;
    const converged = completedSimulations.filter(s => getConvergenceStatus(s) === 'converged').length;
    const totalIterations = completedSimulations.reduce(
      (sum, s) => sum + (s.currentIteration ?? 0), 0
    );
    return { total, converged, totalIterations };
  }, [completedSimulations]);

  if (error) {
    return (
      <div className="flex h-screen overflow-hidden">
        <Sidebar />
        <div className="flex-1 flex flex-col overflow-hidden">
          <Header />
          <DashboardShell>
            <SectionCard>
              <EmptyState
                icon={<BarChart3 className="h-6 w-6" />}
                title="Failed to load results"
                description={error.message}
                action={
                  <Button onClick={() => window.location.reload()}>
                    Retry
                  </Button>
                }
              />
            </SectionCard>
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
            <PageHeader
              title="Results"
              description="View and analyze completed simulation results"
              breadcrumbs={[
                { label: 'Dashboard', href: '/' },
                { label: 'Results' },
              ]}
            />

            {/* Filters */}
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search results..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9"
                />
              </div>
              <Select value={convergenceFilter} onValueChange={setConvergenceFilter}>
                <SelectTrigger className="w-[160px]">
                  <Filter className="h-4 w-4 mr-2" />
                  <SelectValue placeholder="Convergence" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Results</SelectItem>
                  <SelectItem value="converged">Converged</SelectItem>
                  <SelectItem value="partial">Partial</SelectItem>
                  <SelectItem value="diverged">Diverged</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Summary Stats */}
            {isLoading ? (
              <div className="grid gap-4 md:grid-cols-3">
                {Array.from({ length: 3 }).map((_, i) => (
                  <Skeleton key={i} className="h-24 rounded-xl" />
                ))}
              </div>
            ) : (
              <div className="grid gap-4 md:grid-cols-3">
                <div className="rounded-xl border bg-card p-4">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <CheckCircle2 className="h-4 w-4" />
                    Total Results
                  </div>
                  <p className="text-2xl font-bold mt-1">{stats.total}</p>
                </div>
                <div className="rounded-xl border bg-card p-4">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                    Converged
                  </div>
                  <p className="text-2xl font-bold mt-1 text-emerald-600">
                    {stats.converged}
                  </p>
                </div>
                <div className="rounded-xl border bg-card p-4">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <BarChart3 className="h-4 w-4" />
                    Total Iterations
                  </div>
                  <p className="text-2xl font-bold mt-1">
                    {stats.totalIterations >= 1000 
                      ? `${(stats.totalIterations / 1000).toFixed(1)}K`
                      : stats.totalIterations}
                  </p>
                </div>
              </div>
            )}

            {/* Results Grid */}
            {isLoading ? (
              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                {Array.from({ length: 6 }).map((_, i) => (
                  <Skeleton key={i} className="h-48 rounded-xl" />
                ))}
              </div>
            ) : filteredResults.length === 0 ? (
              <SectionCard>
                <EmptyState
                  icon={<BarChart3 className="h-6 w-6" />}
                  title="No results found"
                  description={
                    searchQuery || convergenceFilter !== 'all'
                      ? 'Try adjusting your filters'
                      : 'Complete a simulation to see results here'
                  }
                  action={
                    !searchQuery && convergenceFilter === 'all' && (
                      <Button asChild>
                        <Link href="/simulations/new">Start Simulation</Link>
                      </Button>
                    )
                  }
                />
              </SectionCard>
            ) : (
              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                {filteredResults.map((sim) => (
                  <ResultCard key={sim.id} simulation={sim} />
                ))}
              </div>
            )}
          </div>
        </DashboardShell>
      </div>
    </div>
  );
}

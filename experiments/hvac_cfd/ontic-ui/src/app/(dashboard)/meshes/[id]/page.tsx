/**
 * Mesh Detail Page
 * 
 * View mesh details, 3D visualization, patches, and linked simulations.
 */

'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  Box,
  Download,
  Trash2,
  Play,
  ChevronRight,
  AlertCircle,
  Layers,
  Grid3X3,
  Ruler,
  FileBox,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Sidebar,
  Header,
  DashboardShell,
  SectionCard,
  EmptyState,
} from '@/components/layout';
import { MeshViewer } from '@/components/cfd';
import { useMesh, useDeleteMesh, useSimulations } from '@/hooks';
import { useToast } from '@/components/ui/use-toast';

// ============================================
// PATCH TYPE BADGE
// ============================================

function PatchTypeBadge({ type }: { type: string }) {
  const config: Record<string, 'default' | 'secondary' | 'outline' | 'destructive'> = {
    inlet: 'default',
    outlet: 'secondary',
    wall: 'outline',
    symmetry: 'outline',
    periodic: 'secondary',
    empty: 'outline',
  };

  return (
    <Badge variant={config[type] ?? 'outline'} className="capitalize">
      {type}
    </Badge>
  );
}

// ============================================
// 3D MESH PREVIEW
// ============================================

function MeshPreview({ mesh }: { mesh: NonNullable<ReturnType<typeof useMesh>['data']> }) {
  return (
    <div className="h-80">
      <MeshViewer mesh={mesh} className="h-full" />
    </div>
  );
}

// ============================================
// MAIN PAGE
// ============================================

export default function MeshDetailPage({
  params,
}: {
  params: { id: string };
}) {
  const { id } = params;
  const router = useRouter();
  const { toast } = useToast();
  
  const { data: mesh, isLoading, error } = useMesh(id);
  const { data: simulations } = useSimulations();
  const deleteMesh = useDeleteMesh();

  // Filter simulations that use this mesh
  const linkedSimulations = (simulations ?? []).filter(s => s.meshId === id);

  const handleDelete = async () => {
    if (!confirm('Are you sure you want to delete this mesh?')) return;
    try {
      await deleteMesh.mutateAsync(id);
      toast({ title: 'Mesh deleted' });
      router.push('/meshes');
    } catch (err) {
      toast({ title: 'Delete failed', description: (err as Error).message, variant: 'destructive' });
    }
  };

  const formatCellCount = (count: number) => {
    if (count >= 1e6) return `${(count / 1e6).toFixed(2)}M`;
    if (count >= 1e3) return `${(count / 1e3).toFixed(1)}K`;
    return count.toString();
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
                title="Mesh not found"
                description={error.message}
                action={
                  <Button asChild>
                    <Link href="/meshes">Back to Meshes</Link>
                  </Button>
                }
              />
            </SectionCard>
          </DashboardShell>
        </div>
      </div>
    );
  }

  if (isLoading || !mesh) {
    return (
      <div className="flex h-screen overflow-hidden">
        <Sidebar />
        <div className="flex-1 flex flex-col overflow-hidden">
          <Header />
          <DashboardShell>
            <div className="space-y-6">
              <Skeleton className="h-20" />
              <div className="grid gap-6 lg:grid-cols-3">
                <Skeleton className="h-80 lg:col-span-2" />
                <Skeleton className="h-80" />
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
                  <Link href="/meshes" className="hover:text-foreground">Meshes</Link>
                  <ChevronRight className="h-4 w-4" />
                  <span>{mesh.name}</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 rounded-lg bg-purple-500/10 flex items-center justify-center">
                    <Box className="h-5 w-5 text-purple-500" />
                  </div>
                  <div>
                    <h1 className="text-2xl font-bold">{mesh.name}</h1>
                    {mesh.description && (
                      <p className="text-muted-foreground">{mesh.description}</p>
                    )}
                  </div>
                </div>
              </div>
              <div className="flex gap-2">
                <Button variant="outline" asChild>
                  <Link href={`/simulations/new?meshId=${id}`}>
                    <Play className="h-4 w-4 mr-2" />
                    New Simulation
                  </Link>
                </Button>
                <Button variant="outline">
                  <Download className="h-4 w-4 mr-2" />
                  Export
                </Button>
              </div>
            </div>

            {/* Stats Cards */}
            <div className="grid gap-4 md:grid-cols-4">
              <div className="rounded-xl border bg-card p-4">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Grid3X3 className="h-4 w-4" />
                  Cells
                </div>
                <p className="text-2xl font-bold mt-1">{formatCellCount(mesh.cellCount)}</p>
              </div>
              <div className="rounded-xl border bg-card p-4">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Layers className="h-4 w-4" />
                  Patches
                </div>
                <p className="text-2xl font-bold mt-1">{mesh.patchCount}</p>
              </div>
              <div className="rounded-xl border bg-card p-4">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Ruler className="h-4 w-4" />
                  Domain Size
                </div>
                <p className="text-lg font-bold mt-1">
                  {mesh.domain_size?.map(d => d.toFixed(1)).join(' × ') ?? 'N/A'} m
                </p>
              </div>
              <div className="rounded-xl border bg-card p-4">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <FileBox className="h-4 w-4" />
                  Resolution
                </div>
                <p className="text-lg font-bold mt-1">
                  {mesh.resolution?.nx ?? 0} × {mesh.resolution?.ny ?? 0} × {mesh.resolution?.nz ?? 0}
                </p>
              </div>
            </div>

            {/* Main Content */}
            <Tabs defaultValue="preview" className="space-y-6">
              <TabsList>
                <TabsTrigger value="preview">Preview</TabsTrigger>
                <TabsTrigger value="patches">Patches ({mesh.patches?.length ?? 0})</TabsTrigger>
                <TabsTrigger value="simulations">Simulations ({linkedSimulations.length})</TabsTrigger>
              </TabsList>

              <TabsContent value="preview">
                <SectionCard title="3D Preview">
                  <MeshPreview mesh={mesh} />
                </SectionCard>
              </TabsContent>

              <TabsContent value="patches">
                <SectionCard title="Boundary Patches" noPadding>
                  {mesh.patches && mesh.patches.length > 0 ? (
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Name</TableHead>
                          <TableHead>Type</TableHead>
                          <TableHead>Faces</TableHead>
                          <TableHead>Velocity</TableHead>
                          <TableHead>Pressure</TableHead>
                          <TableHead>Temperature</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {mesh.patches.map((patch) => (
                          <TableRow key={patch.id ?? patch.name}>
                            <TableCell className="font-medium">{patch.name}</TableCell>
                            <TableCell><PatchTypeBadge type={patch.type} /></TableCell>
                            <TableCell>{patch.faceCount?.toLocaleString() ?? '-'}</TableCell>
                            <TableCell>
                              {patch.velocity 
                                ? `[${patch.velocity.map(v => v.toFixed(1)).join(', ')}]` 
                                : '-'}
                            </TableCell>
                            <TableCell>
                              {patch.pressure !== undefined ? `${patch.pressure} Pa` : '-'}
                            </TableCell>
                            <TableCell>
                              {patch.temperature !== undefined ? `${patch.temperature} K` : '-'}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  ) : (
                    <div className="p-8">
                      <EmptyState
                        icon={<Layers className="h-6 w-6" />}
                        title="No patches defined"
                        description="Add boundary patches to configure inlet, outlet, and wall conditions"
                      />
                    </div>
                  )}
                </SectionCard>
              </TabsContent>

              <TabsContent value="simulations">
                <SectionCard title="Linked Simulations">
                  {linkedSimulations.length > 0 ? (
                    <div className="space-y-3">
                      {linkedSimulations.map((sim) => (
                        <Link
                          key={sim.id}
                          href={`/simulations/${sim.id}`}
                          className="flex items-center justify-between p-4 rounded-lg border hover:bg-muted/50 transition-colors"
                        >
                          <div>
                            <p className="font-medium">{sim.name}</p>
                            <p className="text-sm text-muted-foreground">
                              {sim.currentIteration?.toLocaleString() ?? 0} / {sim.maxIterations?.toLocaleString()} iterations
                            </p>
                          </div>
                          <Badge variant={sim.status === 'completed' ? 'default' : sim.status === 'running' ? 'secondary' : 'outline'}>
                            {sim.status}
                          </Badge>
                        </Link>
                      ))}
                    </div>
                  ) : (
                    <EmptyState
                      icon={<Play className="h-6 w-6" />}
                      title="No simulations"
                      description="Create a simulation using this mesh"
                      action={
                        <Button asChild>
                          <Link href={`/simulations/new?meshId=${id}`}>
                            New Simulation
                          </Link>
                        </Button>
                      }
                    />
                  )}
                </SectionCard>
              </TabsContent>
            </Tabs>

            {/* Danger Zone */}
            <SectionCard title="Danger Zone" className="border-destructive/50">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Delete Mesh</p>
                  <p className="text-sm text-muted-foreground">
                    Permanently remove this mesh. Cannot delete if in use by simulations.
                  </p>
                </div>
                <Button
                  variant="destructive"
                  onClick={handleDelete}
                  disabled={deleteMesh.isPending || linkedSimulations.length > 0}
                >
                  <Trash2 className="h-4 w-4 mr-2" />
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

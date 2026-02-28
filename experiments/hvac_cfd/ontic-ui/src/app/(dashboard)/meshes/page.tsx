/**
 * Meshes Page
 * 
 * View, import, and manage computational meshes.
 */

'use client';

import { useState, useCallback } from 'react';
import Link from 'next/link';
import {
  Plus,
  Search,
  Upload,
  LayoutGrid,
  List,
  Box,
  MoreHorizontal,
  Trash2,
  Eye,
  Download,
  Copy,
  FileUp,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Skeleton } from '@/components/ui/skeleton';
import { Progress } from '@/components/ui/progress';
import {
  Sidebar,
  Header,
  DashboardShell,
  PageHeader,
  SectionCard,
  EmptyState,
} from '@/components/layout';
import { useMeshes, useUploadMesh, useDeleteMesh } from '@/hooks';
import { useToast } from '@/components/ui/use-toast';
import type { MeshSummary } from '@/types';

// ============================================
// MESH UPLOAD DIALOG
// ============================================

function MeshUploadDialog() {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isOpen, setIsOpen] = useState(false);
  const { toast } = useToast();
  const uploadMesh = useUploadMesh();

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    handleFiles(files);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleFiles = async (files: File[]) => {
    if (files.length === 0) return;
    
    setUploadProgress(0);
    
    // Upload each file
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (!file) continue;
      
      setUploadProgress(Math.round((i / files.length) * 100));
      
      try {
        await uploadMesh.mutateAsync(file);
        toast({
          title: 'Mesh uploaded',
          description: `Successfully imported ${file.name}`,
        });
      } catch (error) {
        toast({
          title: 'Upload failed',
          description: error instanceof Error ? error.message : 'Unknown error',
          variant: 'destructive',
        });
      }
    }
    
    setUploadProgress(100);
    setTimeout(() => {
      setIsOpen(false);
      setUploadProgress(0);
    }, 500);
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button>
          <Upload className="h-4 w-4 mr-2" />
          Import Mesh
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Import Mesh</DialogTitle>
          <DialogDescription>
            Upload OpenFOAM polyMesh, STL, or CGNS files
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 pt-4">
          <div
            className={`
              border-2 border-dashed rounded-lg p-8 text-center transition-colors
              ${isDragging ? 'border-primary bg-primary/5' : 'border-muted-foreground/25'}
              ${uploadMesh.isPending ? 'pointer-events-none' : 'cursor-pointer hover:border-primary/50'}
            `}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => {
              if (!uploadMesh.isPending) {
                document.getElementById('mesh-upload')?.click();
              }
            }}
          >
            <input
              id="mesh-upload"
              type="file"
              className="hidden"
              accept=".stl,.cgns,.zip"
              multiple
              onChange={(e) => {
                if (e.target.files) {
                  handleFiles(Array.from(e.target.files));
                }
              }}
            />
            {uploadMesh.isPending ? (
              <div className="space-y-3">
                <div className="flex items-center justify-center">
                  <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
                    <FileUp className="h-6 w-6 text-primary animate-pulse" />
                  </div>
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-medium">Uploading...</p>
                  <Progress value={uploadProgress} className="h-2 w-48 mx-auto" />
                  <p className="text-xs text-muted-foreground">{uploadProgress}%</p>
                </div>
              </div>
            ) : (
              <>
                <div className="flex items-center justify-center mb-4">
                  <div className="h-12 w-12 rounded-full bg-muted flex items-center justify-center">
                    <Upload className="h-6 w-6 text-muted-foreground" />
                  </div>
                </div>
                <p className="text-sm font-medium">
                  Drag and drop your mesh files here
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  or click to browse
                </p>
                <div className="flex gap-2 justify-center mt-4">
                  <Badge variant="outline">STL</Badge>
                  <Badge variant="outline">OpenFOAM</Badge>
                  <Badge variant="outline">CGNS</Badge>
                </div>
              </>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

// ============================================
// MESH CARD
// ============================================

interface MeshCardProps {
  mesh: MeshSummary;
  onDelete: (id: string) => void;
  isDeleting: boolean;
}

function MeshCard({ mesh, onDelete, isDeleting }: MeshCardProps) {
  const formatCellCount = (count: number) => {
    if (count >= 1e6) return `${(count / 1e6).toFixed(1)}M`;
    if (count >= 1e3) return `${(count / 1e3).toFixed(1)}K`;
    return count.toString();
  };

  return (
    <div className="rounded-xl border bg-card p-4 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-lg bg-purple-500/10 flex items-center justify-center">
            <Box className="h-5 w-5 text-purple-500" />
          </div>
          <div>
            <h3 className="font-medium">{mesh.name}</h3>
            <p className="text-xs text-muted-foreground">
              {formatCellCount(mesh.cellCount)} cells
            </p>
          </div>
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" className="h-8 w-8">
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem asChild>
              <Link href={`/meshes/${mesh.id}`}>
                <Eye className="h-4 w-4 mr-2" />
                View
              </Link>
            </DropdownMenuItem>
            <DropdownMenuItem>
              <Copy className="h-4 w-4 mr-2" />
              Duplicate
            </DropdownMenuItem>
            <DropdownMenuItem>
              <Download className="h-4 w-4 mr-2" />
              Export
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem 
              className="text-destructive"
              onClick={() => onDelete(mesh.id)}
              disabled={isDeleting}
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      <div className="mt-4 grid grid-cols-3 gap-2 text-center">
        <div className="rounded-lg bg-muted/50 p-2">
          <p className="text-xs text-muted-foreground">Patches</p>
          <p className="font-medium">{mesh.patchCount}</p>
        </div>
        <div className="rounded-lg bg-muted/50 p-2">
          <p className="text-xs text-muted-foreground">Quality</p>
          <p className="font-medium text-emerald-600">Good</p>
        </div>
        <div className="rounded-lg bg-muted/50 p-2">
          <p className="text-xs text-muted-foreground">Size</p>
          <p className="font-medium text-xs">{(mesh.cellCount * 0.001).toFixed(1)} MB</p>
        </div>
      </div>
    </div>
  );
}

// ============================================
// MAIN PAGE
// ============================================

export default function MeshesPage() {
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const { toast } = useToast();

  const { data: meshes, isLoading, error } = useMeshes();
  const deleteMesh = useDeleteMesh();

  const handleDelete = async (id: string) => {
    try {
      await deleteMesh.mutateAsync(id);
      toast({
        title: 'Mesh deleted',
        description: 'The mesh has been removed.',
      });
    } catch (err) {
      toast({
        title: 'Delete failed',
        description: err instanceof Error ? err.message : 'Unknown error',
        variant: 'destructive',
      });
    }
  };

  const filteredMeshes = (meshes ?? []).filter((mesh) =>
    mesh.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  if (error) {
    return (
      <div className="flex h-screen overflow-hidden">
        <Sidebar />
        <div className="flex-1 flex flex-col overflow-hidden">
          <Header />
          <DashboardShell>
            <SectionCard>
              <EmptyState
                icon={<Box className="h-6 w-6" />}
                title="Failed to load meshes"
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
              title="Meshes"
              description="Manage computational meshes for CFD simulations"
              breadcrumbs={[
                { label: 'Dashboard', href: '/' },
                { label: 'Meshes' },
              ]}
              actions={<MeshUploadDialog />}
            />

            {/* Filters */}
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search meshes..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9"
                />
              </div>
              <div className="flex border rounded-lg">
                <Button
                  variant={viewMode === 'grid' ? 'secondary' : 'ghost'}
                  size="icon"
                  onClick={() => setViewMode('grid')}
                >
                  <LayoutGrid className="h-4 w-4" />
                </Button>
                <Button
                  variant={viewMode === 'list' ? 'secondary' : 'ghost'}
                  size="icon"
                  onClick={() => setViewMode('list')}
                >
                  <List className="h-4 w-4" />
                </Button>
              </div>
            </div>

            {/* Content */}
            {isLoading ? (
              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                {Array.from({ length: 6 }).map((_, i) => (
                  <Skeleton key={i} className="h-40 rounded-xl" />
                ))}
              </div>
            ) : filteredMeshes.length === 0 ? (
              <SectionCard>
                <EmptyState
                  icon={<Box className="h-6 w-6" />}
                  title="No meshes found"
                  description={
                    searchQuery
                      ? 'Try a different search term'
                      : 'Import your first mesh to get started'
                  }
                  action={!searchQuery && <MeshUploadDialog />}
                />
              </SectionCard>
            ) : viewMode === 'grid' ? (
              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                {filteredMeshes.map((mesh) => (
                  <MeshCard 
                    key={mesh.id} 
                    mesh={mesh} 
                    onDelete={handleDelete}
                    isDeleting={deleteMesh.isPending}
                  />
                ))}
              </div>
            ) : (
              <SectionCard noPadding>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Cells</TableHead>
                      <TableHead>Patches</TableHead>
                      <TableHead>Quality</TableHead>
                      <TableHead>Created</TableHead>
                      <TableHead className="w-[80px]">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredMeshes.map((mesh) => (
                      <TableRow key={mesh.id}>
                        <TableCell className="font-medium">{mesh.name}</TableCell>
                        <TableCell>{(mesh.cellCount / 1e6).toFixed(2)}M</TableCell>
                        <TableCell>{mesh.patchCount}</TableCell>
                        <TableCell>
                          <Badge variant="outline" className="text-emerald-600">
                            Good
                          </Badge>
                        </TableCell>
                        <TableCell className="text-muted-foreground">
                          {new Date(mesh.createdAt).toLocaleDateString()}
                        </TableCell>
                        <TableCell>
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button variant="ghost" size="icon" className="h-8 w-8">
                                <MoreHorizontal className="h-4 w-4" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              <DropdownMenuItem asChild>
                                <Link href={`/meshes/${mesh.id}`}>View</Link>
                              </DropdownMenuItem>
                              <DropdownMenuItem>Duplicate</DropdownMenuItem>
                              <DropdownMenuItem>Export</DropdownMenuItem>
                              <DropdownMenuSeparator />
                              <DropdownMenuItem 
                                className="text-destructive"
                                onClick={() => handleDelete(mesh.id)}
                                disabled={deleteMesh.isPending}
                              >
                                Delete
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </SectionCard>
            )}
          </div>
        </DashboardShell>
      </div>
    </div>
  );
}

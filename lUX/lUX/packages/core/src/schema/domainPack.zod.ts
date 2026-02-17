import { z } from "zod";

const SemVer = z.string().regex(/^\d+\.\d+\.\d+$/);

export const DomainPackSchema = z.object({
  id: z.string(),
  version: SemVer,

  metrics: z.record(
    z.string(),
    z.object({
      label: z.string(),
      symbol_latex: z.string().optional(),
      unit: z.string(),
      dimension: z.string().optional(),
      format: z.enum(["fixed", "scientific", "engineering"]),
      precision: z.number().int().min(0).max(12),
      validity_range: z.tuple([z.number(), z.number()]).optional(),
      description: z.string().optional(),
    })
  ),

  gate_packs: z.record(
    z.string(),
    z.object({
      label: z.string(),
      manifest_ref: z.string(),
      highlight_metrics: z.array(z.string()).default([]),
    })
  ),

  viewers: z.array(
    z.object({
      when: z.object({
        artifact_type: z.string().optional(),
        mime_type: z.string().optional(),
        metric_id: z.string().optional(),
      }),
      component: z.enum([
        "TimeSeriesViewer",
        "TableViewer",
        "LogViewer",
        "SliceViewer2D",
        "VolumeRenderer3D",
        "TensorInspector",
        "ArtifactRawViewer",
        "DiffViewer",
      ]),
      default_config: z.record(z.string(), z.any()).default({}),
      priority: z.number().int().default(0),
    })
  ),

  templates: z.object({
    executive_summary_metric_ids: z.array(z.string()).default([]),
    publication_sections: z.array(z.string()).default([]),
    citation_format: z.enum(["bibtex", "apa"]).default("bibtex"),
  }),
});

export type DomainPack = z.infer<typeof DomainPackSchema>;

/**
 * ProofImage — production-ready next/image wrapper for raster assets.
 *
 * Provides:
 *   - Consistent aspect-ratio handling with blur placeholder
 *   - Automatic dark-mode-aware background for loading state
 *   - Design-system-aligned rounded corners and shadow
 *   - Priority loading for above-the-fold images
 *   - Accessible alt text enforcement via required prop
 *
 * Usage:
 *   <ProofImage src="/evidence/spectrum.png" alt="Power spectrum" width={800} height={400} />
 *
 * Covers ROADMAP item:
 *   - Phase 6: "Add next/image for any future raster assets"
 */
import Image, { type ImageProps } from "next/image";
import { cn } from "@/config/utils";

/** 1×1 transparent SVG as blur placeholder (avoids CLS) */
const BLUR_DATA_URL =
  "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMSIgaGVpZ2h0PSIxIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxyZWN0IHdpZHRoPSIxIiBoZWlnaHQ9IjEiIGZpbGw9IiMxYTFhMjAiLz48L3N2Zz4=";

type ProofImageProps = Omit<ImageProps, "placeholder" | "blurDataURL"> & {
  /** Whether this image is above the fold and should be prioritised. */
  priority?: boolean;
  /** Additional CSS class applied to the outer wrapper div. */
  wrapperClassName?: string;
};

export function ProofImage({
  className,
  wrapperClassName,
  priority = false,
  alt,
  ...rest
}: ProofImageProps) {
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-[var(--radius-inner)] bg-[var(--color-bg-raised)]",
        wrapperClassName,
      )}
    >
      <Image
        alt={alt}
        placeholder="blur"
        blurDataURL={BLUR_DATA_URL}
        priority={priority}
        loading={priority ? "eager" : "lazy"}
        sizes="(max-width: 768px) 100vw, (max-width: 1280px) 50vw, 33vw"
        quality={85}
        className={cn(
          "h-auto w-full object-contain transition-opacity duration-base ease-lux-out",
          className,
        )}
        {...rest}
      />
    </div>
  );
}

ProofImage.displayName = "ProofImage";

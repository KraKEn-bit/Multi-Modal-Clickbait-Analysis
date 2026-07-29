"use client";

import { useEffect, useState } from "react";
import { assetUrl, youtubeStoryboardFrame, youtubeThumb } from "@/lib/api";

/**
 * Loads a cached VTCF frame. Prefers same-origin /cached-frames/ (works on
 * Vercel). Falls back to YouTube storyboard stills, then the video thumbnail.
 */
export function TemporalFrameImage({
  videoId,
  url,
  frameIndex,
  label,
  className = "aspect-video w-full object-cover",
}: {
  videoId: string;
  url: string;
  frameIndex: 0 | 1 | 2;
  label: string;
  className?: string;
}) {
  const primary = assetUrl(url);
  const storyboard = youtubeStoryboardFrame(videoId, frameIndex);
  const thumb = youtubeThumb(videoId);
  const [src, setSrc] = useState(primary);

  useEffect(() => {
    setSrc(primary);
  }, [primary, videoId, frameIndex]);

  return (
    /* eslint-disable-next-line @next/next/no-img-element */
    <img
      src={src}
      alt={label}
      className={className}
      loading="lazy"
      onError={() => {
        setSrc((current) => {
          if (current === primary) return storyboard;
          if (current === storyboard) return thumb;
          return current;
        });
      }}
    />
  );
}

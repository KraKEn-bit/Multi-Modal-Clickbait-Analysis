/**
 * Extract a YouTube video ID from common URL shapes (watch, youtu.be, shorts, embed).
 * Returns null when the input is not a recognizable YouTube video link.
 */
export function extractYoutubeVideoId(input: string): string | null {
  const raw = input.trim();
  if (!raw) return null;

  // Bare 11-char ID
  if (/^[A-Za-z0-9_-]{11}$/.test(raw)) return raw;

  try {
    const withProtocol = /^https?:\/\//i.test(raw) ? raw : `https://${raw}`;
    const url = new URL(withProtocol);
    const host = url.hostname.replace(/^www\./, "").toLowerCase();

    if (host === "youtu.be") {
      const id = url.pathname.split("/").filter(Boolean)[0] ?? "";
      return /^[A-Za-z0-9_-]{11}$/.test(id) ? id : null;
    }

    if (host === "youtube.com" || host === "m.youtube.com" || host === "music.youtube.com") {
      const v = url.searchParams.get("v");
      if (v && /^[A-Za-z0-9_-]{11}$/.test(v)) return v;

      const parts = url.pathname.split("/").filter(Boolean);
      // /shorts/ID, /embed/ID, /live/ID, /v/ID
      if (
        parts.length >= 2 &&
        ["shorts", "embed", "live", "v"].includes(parts[0]) &&
        /^[A-Za-z0-9_-]{11}$/.test(parts[1])
      ) {
        return parts[1];
      }
    }
  } catch {
    /* fall through */
  }

  const match = raw.match(
    /(?:youtube\.com\/(?:watch\?.*v=|shorts\/|embed\/|live\/|v\/)|youtu\.be\/)([A-Za-z0-9_-]{11})/,
  );
  return match?.[1] ?? null;
}

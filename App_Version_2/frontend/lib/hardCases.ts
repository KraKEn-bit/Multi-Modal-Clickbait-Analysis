/**
 * Local hard-case corpus for the sticky-scroll walkthrough.
 * Metrics and labels are inlined; frame paths point at the real
 * hook / context / delivery PNGs extracted by the VTCF pipeline
 * (frame_0 → hook, frame_1 → context, frame_2 → delivery).
 */

export type HardCase = {
  video_id: string;
  title: string;
  verdict: "CLICKBAIT" | "GENUINE";
  confidence: number;
  tds_score: number;
  ground_truth: "CLICKBAIT" | "GENUINE";
  text_only: {
    verdict: "CLICKBAIT" | "GENUINE";
    confidence: number;
    wrong: true;
  };
  alignment_scores: {
    hook: number;
    context: number;
    delivery: number;
  };
  /** Backend-relative paths — resolve with assetUrl() at render time. */
  frame_urls: [string, string, string];
  example_label: string;
  example_description: string;
};

/** Real temporal frames from cached_examples (hook → context → delivery). */
function cachedFrames(videoId: string): [string, string, string] {
  return [
    `/cached-frames/${videoId}/frame_0.png`,
    `/cached-frames/${videoId}/frame_1.png`,
    `/cached-frames/${videoId}/frame_2.png`,
  ];
}

export const HARD_CASES: HardCase[] = [
  {
    video_id: "DhESX8gA7wk",
    title:
      "বড় বোনের বিয়েতে সিনেমা স্টাইলে অপহরন ছোট বোন। অবাক করা এই কাজ করল বর নিজেই। দেখুন তোলপাড় সারাদেশ",
    verdict: "CLICKBAIT",
    confidence: 94.82,
    tds_score: 0.1108,
    ground_truth: "CLICKBAIT",
    text_only: { verdict: "GENUINE", confidence: 98.38, wrong: true },
    alignment_scores: { hook: -0.0353, context: -0.0026, delivery: -0.0356 },
    frame_urls: cachedFrames("DhESX8gA7wk"),
    example_label: "Sensational wedding headline",
    example_description:
      "Title promises a cinema-style kidnapping. Frames never deliver it — text-only called it genuine.",
  },
  {
    video_id: "OoUO4vjgM4c",
    title:
      "তিন দিনে আফগানিস্তানের পাঁচ প্রাদেশিক রাজধানী তালেবানের দখলে || [Taleban Situation]",
    verdict: "GENUINE",
    confidence: 99.96,
    tds_score: 0.8354,
    ground_truth: "GENUINE",
    text_only: { verdict: "CLICKBAIT", confidence: 99.98, wrong: true },
    alignment_scores: { hook: -0.0132, context: -0.0266, delivery: 0.0271 },
    frame_urls: cachedFrames("OoUO4vjgM4c"),
    example_label: "Breaking-news title, factual bulletin",
    example_description:
      "Title-only flagged clickbait. VTCF reads the newsroom visuals and rescues the genuine label.",
  },
  {
    video_id: "pYganyZsHYM",
    title:
      "🚨 A Mysterious Incident in the Jungle! You Won't Believe What Happened Next! 😱🔥",
    verdict: "CLICKBAIT",
    confidence: 92.88,
    tds_score: 0.5792,
    ground_truth: "CLICKBAIT",
    text_only: { verdict: "GENUINE", confidence: 87.4, wrong: true },
    alignment_scores: { hook: -0.0593, context: 0.0005, delivery: -0.0185 },
    frame_urls: cachedFrames("pYganyZsHYM"),
    example_label: "Viral English bait",
    example_description:
      "Sensational English headline on recycled footage. Polished wording fooled the text channel.",
  },
  {
    video_id: "hcFpC8R6c24",
    title: "আজকের শীর্ষ ৭ খবর | Shirsho 7 | 24 July 2026 | Independent TV",
    verdict: "GENUINE",
    confidence: 99.91,
    tds_score: 0.5275,
    ground_truth: "GENUINE",
    text_only: { verdict: "CLICKBAIT", confidence: 76.2, wrong: true },
    alignment_scores: { hook: 0.0689, context: 0.0675, delivery: -0.0019 },
    frame_urls: cachedFrames("hcFpC8R6c24"),
    example_label: "News bulletin — false text alarm",
    example_description:
      "A straightforward TV bulletin. Title-only over-triggered; visual timeline confirms genuine news.",
  },
];

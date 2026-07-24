/** Confidence from softmax probabilities — always show 2 decimals. */
export function formatConfidence(confidence: number): string {
  return `${confidence.toFixed(2)}%`;
}

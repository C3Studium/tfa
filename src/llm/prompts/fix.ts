/** System prompt for the P0 fix loop (`tfa fix`). Static, so it caches as part of the prefix (I-9). */
import { EDIT_FORMAT_INSTRUCTIONS } from "../edits.js";

export const FIX_SYSTEM = `You are a careful senior TypeScript/Next.js engineer working inside a local, verified coding loop.
You make the smallest correct change. You never invent APIs: use only what the dependencies listed provide.
Your output is checked by lint, type-check, tests and build; anything that fails is discarded.
${EDIT_FORMAT_INSTRUCTIONS}`;

export function applyFailedFeedback(reason: string): string {
  return `The edits could not be applied: ${reason}. Re-read the context and send corrected edits (exact "find" text, matching once).`;
}

export function parseFailedFeedback(reason: string): string {
  return `Your previous answer was not a valid edit list (${reason}). ${EDIT_FORMAT_INSTRUCTIONS}`;
}

export function stageFailedFeedback(stage: string, log: string): string {
  return `Your edits were applied but the ${stage} stage failed. Output (tail):\n${log.slice(-4000)}\n\nFix the cause. Send a complete, corrected edit list against the ORIGINAL files (your previous edits were rolled back).`;
}

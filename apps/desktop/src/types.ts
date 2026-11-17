export type TranslationStatus = 'NEW' | 'DRAFT' | 'REVIEW' | 'COMMITTED';

export interface TranslationUnit {
  id: number;
  unit_identifier: string;
  source_text: string;
  target_text: string | null;
  speaker: string | null;
  status: TranslationStatus | string;
  row_order: number;
}

export interface GridStats {
  totalUnits: number;
  renderedNodes: number;
  activeChunkCount: number;
  queryLatencyMs: number;
}

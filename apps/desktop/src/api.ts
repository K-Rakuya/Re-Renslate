import { invoke, isTauri } from '@tauri-apps/api/core';
import { TranslationUnit } from './types';

// Standalone mock generator for testing 300,000 rows in browser/preview mode
const MOCK_TOTAL_ROWS = 300_000;

export async function openProjectDb(path: string): Promise<string> {
  if (!isTauri()) {
    return 'Mock Database opened';
  }
  return await invoke<string>('open_project_db', { path });
}

export async function closeProjectDb(): Promise<void> {
  if (!isTauri()) return;
  await invoke('close_project_db');
}

export async function getTotalCount(): Promise<number> {
  if (!isTauri()) {
    return MOCK_TOTAL_ROWS;
  }
  return await invoke<number>('get_total_count');
}

export async function fetchTranslationPage(
  cursorRowOrder: number,
  limit: number = 200
): Promise<TranslationUnit[]> {
  if (!isTauri()) {
    // Generate synthetic rows starting after cursorRowOrder
    const rows: TranslationUnit[] = [];
    const startOrder = cursorRowOrder + 1;
    const endOrder = Math.min(startOrder + limit - 1, MOCK_TOTAL_ROWS);
    for (let order = startOrder; order <= endOrder; order++) {
      rows.push({
        id: order,
        unit_identifier: `line_${order.toString().padStart(6, '0')}`,
        source_text: `Original dialogue text for story segment #${order}. Eileen smiles warmly.`,
        target_text: order % 3 === 0 ? `Lời thoại tiếng Việt cho đoạn truyện #${order}. Eileen mỉm cười ấm áp.` : null,
        speaker: order % 2 === 0 ? 'eileen' : 'narrator',
        status: order % 3 === 0 ? 'COMMITTED' : 'NEW',
        row_order: order,
      });
    }
    return rows;
  }
  return await invoke<TranslationUnit[]>('fetch_translation_page', {
    cursorRowOrder,
    limit,
  });
}

export async function searchUnits(
  query: string,
  limit: number = 200
): Promise<TranslationUnit[]> {
  if (!isTauri()) {
    const q = query.toLowerCase();
    const rows: TranslationUnit[] = [];
    for (let i = 1; i <= 200; i++) {
      rows.push({
        id: i * 100,
        unit_identifier: `search_${i}`,
        source_text: `Found result for "${q}" in dialogue line #${i * 100}`,
        target_text: `Kết quả tìm kiếm cho "${q}" tại dòng #${i * 100}`,
        speaker: 'eileen',
        status: 'COMMITTED',
        row_order: i * 100,
      });
    }
    return rows;
  }
  return await invoke<TranslationUnit[]>('search_units', {
    query,
    limit,
  });
}

export async function callSidecarRpc(method: string, payload: unknown): Promise<unknown> {
  if (!isTauri()) {
    return { ok: true, mock: true };
  }
  return await invoke('call_sidecar_rpc', { method, payload });
}

export async function getSidecarStatus(): Promise<boolean> {
  if (!isTauri()) return true;
  return await invoke<boolean>('get_sidecar_status');
}

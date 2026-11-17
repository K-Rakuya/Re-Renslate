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
    throw new Error('Chức năng RPC yêu cầu ứng dụng chạy trong môi trường Desktop Tauri (không hỗ trợ chạy thử nghiệm trên trình duyệt thuần).');
  }
  return await invoke('call_sidecar_rpc', { method, payload });
}

export async function getSidecarStatus(): Promise<boolean> {
  if (!isTauri()) return false;
  return await invoke<boolean>('get_sidecar_status');
}

export interface ScanResult {
  total_extracted: number;
  db_path: string;
  files_scanned: number;
  rpyc_count: number;
}

export interface FontAuditItem {
  font_name: string;
  total_required: number;
  supported_count: number;
  missing_chars: string;
  is_fully_compatible: boolean;
}

export interface FontAuditProjectResult {
  fonts: FontAuditItem[];
  is_all_compatible: boolean;
  fontgroup_dsl: string;
  total_fonts_checked: number;
}

export interface ExportResult {
  exported_files: string[];
  output_dir: string;
  total_items: number;
}

export interface DetectedModel {
  path: string;
  name: string;
  size_mb: number;
}

export interface SystemSpecs {
  gpu_name: string;
  total_vram_mb: number;
  free_vram_mb: number;
  used_vram_mb: number;
  total_ram_mb: number;
  free_ram_mb: number;
  swap_total_mb: number;
  swap_free_mb: number;
  cuda_available: boolean;
  gpu_offload_supported: boolean;
  detected_models: DetectedModel[];
}

export interface ModelAssessment {
  status: 'safe_full_gpu' | 'safe_hybrid' | 'warning_tight' | 'oom_risk' | 'error';
  message: string;
  can_fit: boolean;
  recommended_gpu_layers: number;
  recommended_ctx: number;
  file_size_mb?: number;
  free_vram_mb?: number;
  free_ram_mb?: number;
}

export interface TranslationProgress {
  is_running: boolean;
  total: number;
  current: number;
  current_source: string;
  current_target: string;
  status: string;
  error?: string | null;
}

export interface StartTranslationParams {
  db_path: string;
  engine?: string;
  model_path?: string;
  server_url?: string;
  n_gpu_layers?: number;
  n_ctx?: number;
  temperature?: number;
  limit?: number;
}

export interface ConnectionTestResult {
  online: boolean;
  message: string;
  models?: string[];
}

export async function testInferenceConnection(serverUrl: string): Promise<ConnectionTestResult> {
  const res = await callSidecarRpc('test_inference_connection', { server_url: serverUrl });
  return res as ConnectionTestResult;
}

export async function getSystemSpecs(): Promise<SystemSpecs> {
  const res = await callSidecarRpc('get_system_specs', {});
  return res as SystemSpecs;
}

export async function assessModel(modelPath: string): Promise<ModelAssessment> {
  const res = await callSidecarRpc('assess_model', { model_path: modelPath });
  return res as ModelAssessment;
}

export async function getTranslationProgress(): Promise<TranslationProgress> {
  const res = await callSidecarRpc('get_translation_progress', {});
  return res as TranslationProgress;
}

export async function unloadModel(): Promise<{ status: string }> {
  const res = await callSidecarRpc('unload_model', {});
  return res as { status: string };
}

export async function scanProject(gameDir: string, dbPath?: string): Promise<ScanResult> {
  const res = await callSidecarRpc('scan_project', { game_dir: gameDir, db_path: dbPath });
  return res as ScanResult;
}

export async function auditProjectFonts(gameDir: string, fallbackFont?: string): Promise<FontAuditProjectResult> {
  const res = await callSidecarRpc('audit_project_fonts', { game_dir: gameDir, fallback_font: fallbackFont });
  return res as FontAuditProjectResult;
}

export async function startTranslation(params: StartTranslationParams): Promise<{ translated_count: number; status: string }> {
  const res = await callSidecarRpc('start_translation', params);
  return res as { translated_count: number; status: string };
}

export async function exportPackage(gameDir: string, dbPath?: string, language: string = 'vietnamese'): Promise<ExportResult> {
  const res = await callSidecarRpc('export_package', { game_dir: gameDir, db_path: dbPath, language });
  return res as ExportResult;
}

export async function launchGame(gameDir: string): Promise<{ launched: boolean; executable: string; pid: number }> {
  const res = await callSidecarRpc('launch_game', { game_dir: gameDir });
  return res as { launched: boolean; executable: string; pid: number };
}

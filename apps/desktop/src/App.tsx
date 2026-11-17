import React, { useState, useEffect } from 'react';
import { VirtualGrid } from './VirtualGrid';
import {
  getTotalCount,
  getSidecarStatus,
  openProjectDb,
  scanProject,
  auditProjectFonts,
  startTranslation,
  exportPackage,
  launchGame,
  FontAuditItem,
} from './api';
import {
  FolderOpen,
  Search,
  Layers,
  Sparkles,
  Type,
  FileCheck,
  Play,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
} from 'lucide-react';

type Tab = 'project' | 'editor' | 'ai' | 'font' | 'export';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<Tab>('project');
  const [gameDir, setGameDir] = useState<string>('tests/fixtures');
  const [dbPath, setDbPath] = useState<string>('');
  const [totalCount, setTotalCount] = useState<number>(0);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [isSidecarAlive, setIsSidecarAlive] = useState<boolean>(true);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>('');

  // Scan state
  const [scanStats, setScanStats] = useState<{
    total_extracted: number;
    files_scanned: number;
    rpyc_count: number;
  } | null>(null);

  // Font audit state
  const [fontAuditList, setFontAuditList] = useState<FontAuditItem[]>([]);
  const [fontGroupDsl, setFontGroupDsl] = useState<string>('');
  const [isAllFontsCompatible, setIsAllFontsCompatible] = useState<boolean>(true);

  // Translation state
  const [aiEngine, setAiEngine] = useState<string>('llama.cpp');
  const [translatedProgress, setTranslatedProgress] = useState<number>(0);

  // Export state
  const [exportedFiles, setExportedFiles] = useState<string[]>([]);
  const [exportOutputDir, setExportOutputDir] = useState<string>('');

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const alive = await getSidecarStatus();
        setIsSidecarAlive(alive);
      } catch (e) {
        console.warn(e);
      }
    };
    checkStatus();
  }, []);

  // 1. Scan Project
  const handleScanProject = async () => {
    if (!gameDir.trim()) return;
    setIsLoading(true);
    setStatusMessage('Đang phân tích cấu trúc game, gắn VFS đọc .rpa và trích xuất AST...');
    try {
      const res = await scanProject(gameDir);
      setScanStats({
        total_extracted: res.total_extracted,
        files_scanned: res.files_scanned,
        rpyc_count: res.rpyc_count,
      });
      setDbPath(res.db_path);
      await openProjectDb(res.db_path);
      const count = await getTotalCount();
      setTotalCount(count);
      setStatusMessage(`Quét thành công! Đã trích xuất ${res.total_extracted} câu thoại vào cơ sở dữ liệu.`);
      // Auto switch to Editor tab after scan
      setTimeout(() => setActiveTab('editor'), 1000);
    } catch (err: any) {
      setStatusMessage(`Lỗi khi quét: ${err.message || err}`);
    } finally {
      setIsLoading(false);
    }
  };

  // 2. Audit Fonts
  const handleAuditFonts = async () => {
    if (!gameDir.trim()) return;
    setIsLoading(true);
    setStatusMessage('Đang kiểm toán bảng cmap của các font đối chiếu với 134 ký tự tiếng Việt NFC...');
    try {
      const res = await auditProjectFonts(gameDir);
      setFontAuditList(res.fonts);
      setIsAllFontsCompatible(res.is_all_compatible);
      setFontGroupDsl(res.fontgroup_dsl);
      setStatusMessage(`Đã kiểm tra ${res.total_fonts_checked} tệp font.`);
    } catch (err: any) {
      setStatusMessage(`Lỗi kiểm toán font: ${err.message || err}`);
    } finally {
      setIsLoading(false);
    }
  };

  // 3. Start AI Translation
  const handleStartAi = async () => {
    if (!dbPath.trim()) return;
    setIsLoading(true);
    setStatusMessage(`Đang chạy dịch tự động qua động cơ ${aiEngine} kết hợp TEaR + MBR...`);
    try {
      const res = await startTranslation(dbPath, 500);
      setTranslatedProgress(res.translated_count);
      const count = await getTotalCount();
      setTotalCount(count);
      setStatusMessage(`Đã hoàn tất dịch lô ${res.translated_count} câu thoại!`);
    } catch (err: any) {
      setStatusMessage(`Lỗi dịch: ${err.message || err}`);
    } finally {
      setIsLoading(false);
    }
  };

  // 4. Export Package
  const handleExport = async () => {
    if (!gameDir.trim()) return;
    setIsLoading(true);
    setStatusMessage('Đang xuất gói dịch chuẩn game/tl/vietnamese/ (4 spaces, 0 tabs)...');
    try {
      const res = await exportPackage(gameDir, dbPath || undefined, 'vietnamese');
      setExportedFiles(res.exported_files);
      setExportOutputDir(res.output_dir);
      setStatusMessage(`Xuất thành công ${res.total_items} câu thoại vào ${res.output_dir}!`);
    } catch (err: any) {
      setStatusMessage(`Lỗi xuất gói: ${err.message || err}`);
    } finally {
      setIsLoading(false);
    }
  };

  // 5. Launch Game
  const handleLaunchGame = async () => {
    if (!gameDir.trim()) return;
    try {
      const res = await launchGame(gameDir);
      setStatusMessage(`Đã khởi chạy game thành công: ${res.executable} (PID: ${res.pid})`);
    } catch (err: any) {
      setStatusMessage(`Không thể khởi chạy game: ${err.message || err}`);
    }
  };

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-[#1e1e2e] text-[#cdd6f4] font-sans">
      {/* Top Main Navigation Header */}
      <header className="flex items-center justify-between px-6 py-3 bg-[#181825] border-b border-[#313244] shadow-md z-10">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-9 h-9 rounded-xl bg-gradient-to-tr from-[#89b4fa] via-[#b4befe] to-[#cba6f7] text-[#11111b] font-black text-xl shadow-lg shadow-[#89b4fa]/20">
            R
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold tracking-tight text-[#cdd6f4]">Renslate</h1>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-[#89b4fa]/10 text-[#89b4fa] border border-[#89b4fa]/20">
                Tauri v2 • Closed Loop
              </span>
            </div>
            <p className="text-[11px] text-[#a6adc8]">Hệ thống bản địa hóa Ren'Py tự hành Local-First</p>
          </div>
        </div>

        {/* Workflow Stage Tabs */}
        <nav className="flex items-center p-1 bg-[#11111b] rounded-xl border border-[#313244]/60 gap-1">
          <button
            onClick={() => setActiveTab('project')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'project'
                ? 'bg-[#89b4fa] text-[#11111b] shadow-sm font-semibold'
                : 'text-[#a6adc8] hover:text-[#cdd6f4] hover:bg-[#313244]/40'
            }`}
          >
            <FolderOpen className="w-3.5 h-3.5" />
            1. Dự Án (Project)
          </button>
          <button
            onClick={() => setActiveTab('editor')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'editor'
                ? 'bg-[#89b4fa] text-[#11111b] shadow-sm font-semibold'
                : 'text-[#a6adc8] hover:text-[#cdd6f4] hover:bg-[#313244]/40'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            2. Hội Thoại (Editor)
          </button>
          <button
            onClick={() => setActiveTab('ai')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'ai'
                ? 'bg-[#89b4fa] text-[#11111b] shadow-sm font-semibold'
                : 'text-[#a6adc8] hover:text-[#cdd6f4] hover:bg-[#313244]/40'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            3. Dịch AI (Translate)
          </button>
          <button
            onClick={() => setActiveTab('font')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'font'
                ? 'bg-[#89b4fa] text-[#11111b] shadow-sm font-semibold'
                : 'text-[#a6adc8] hover:text-[#cdd6f4] hover:bg-[#313244]/40'
            }`}
          >
            <Type className="w-3.5 h-3.5" />
            4. Font & Bố Cục
          </button>
          <button
            onClick={() => setActiveTab('export')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'export'
                ? 'bg-[#89b4fa] text-[#11111b] shadow-sm font-semibold'
                : 'text-[#a6adc8] hover:text-[#cdd6f4] hover:bg-[#313244]/40'
            }`}
          >
            <FileCheck className="w-3.5 h-3.5" />
            5. Xuất Bản & Chơi
          </button>
        </nav>

        {/* Sidecar IPC Status */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-lg bg-[#11111b] border border-[#313244]">
            <span className={`w-2 h-2 rounded-full ${isSidecarAlive ? 'bg-[#a6e3a1] animate-pulse' : 'bg-[#f38ba8]'}`} />
            <span className="text-[11px] font-medium text-[#bac2de]">
              {isSidecarAlive ? 'Sidecar IPC Active' : 'Disconnected'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Tab Views */}
      <main className="flex-1 overflow-hidden flex flex-col relative">
        {/* TAB 1: PROJECT SETUP & SCAN */}
        {activeTab === 'project' && (
          <div className="flex-1 overflow-y-auto p-8 max-w-4xl mx-auto w-full flex flex-col justify-center gap-6">
            <div className="bg-[#181825] border border-[#313244] rounded-2xl p-6 shadow-xl flex flex-col gap-5">
              <div className="flex items-center gap-3 border-b border-[#313244] pb-4">
                <FolderOpen className="w-6 h-6 text-[#89b4fa]" />
                <div>
                  <h2 className="text-base font-bold text-[#cdd6f4]">Bước 1: Chọn Thư Mục Game Ren'Py</h2>
                  <p className="text-xs text-[#a6adc8]">Hệ thống sẽ gắn VFS đọc trực tiếp kho nén .rpa, .rpyc và giải nén AST vào SQLite.</p>
                </div>
              </div>

              <div className="flex flex-col gap-2">
                <label className="text-xs font-semibold text-[#bac2de]">Đường dẫn thư mục Game Ren'Py</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={gameDir}
                    onChange={(e) => setGameDir(e.target.value)}
                    placeholder="/path/to/RenpyGame..."
                    className="flex-1 bg-[#11111b] border border-[#313244] rounded-xl px-4 py-2.5 text-xs text-[#cdd6f4] focus:outline-none focus:border-[#89b4fa] transition-colors"
                  />
                  <button
                    onClick={handleScanProject}
                    disabled={isLoading}
                    className="flex items-center gap-2 px-5 py-2.5 bg-[#89b4fa] hover:bg-[#b4befe] text-[#11111b] text-xs font-bold rounded-xl transition-all shadow-md shadow-[#89b4fa]/20 disabled:opacity-50"
                  >
                    <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                    Quét & Trích Xuất Dữ Liệu
                  </button>
                </div>
              </div>

              {scanStats && (
                <div className="grid grid-cols-3 gap-4 mt-2">
                  <div className="bg-[#11111b] border border-[#313244] p-4 rounded-xl text-center">
                    <span className="text-[11px] text-[#a6adc8]">Tổng tệp quét được</span>
                    <p className="text-xl font-bold text-[#89b4fa] mt-1">{scanStats.files_scanned}</p>
                  </div>
                  <div className="bg-[#11111b] border border-[#313244] p-4 rounded-xl text-center">
                    <span className="text-[11px] text-[#a6adc8]">Tệp kịch bản .rpyc</span>
                    <p className="text-xl font-bold text-[#cba6f7] mt-1">{scanStats.rpyc_count}</p>
                  </div>
                  <div className="bg-[#11111b] border border-[#313244] p-4 rounded-xl text-center">
                    <span className="text-[11px] text-[#a6adc8]">Câu thoại trích xuất</span>
                    <p className="text-xl font-bold text-[#a6e3a1] mt-1">{scanStats.total_extracted}</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 2: DIALOGUE EDITOR (VIRTUAL GRID 300K) */}
        {activeTab === 'editor' && (
          <div className="flex-1 flex flex-col h-full overflow-hidden">
            {/* Editor Sub-header with FTS5 search */}
            <div className="flex items-center justify-between px-6 py-2.5 bg-[#181825] border-b border-[#313244]">
              <div className="flex items-center gap-3">
                <span className="text-xs font-semibold text-[#cdd6f4]">
                  Tổng số câu thoại: <span className="text-[#89b4fa] font-bold">{totalCount.toLocaleString()}</span>
                </span>
                <span className="text-xs text-[#585b70]">•</span>
                <span className="text-xs text-[#a6adc8]">VirtualGrid 60fps (Keyset Pagination)</span>
              </div>
              <div className="relative w-96">
                <Search className="absolute left-3 top-2.5 w-4 h-4 text-[#6c7086]" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Tìm kiếm FTS5 unicode61 (nguồn, dịch, người nói)..."
                  className="w-full bg-[#11111b] border border-[#313244] rounded-lg pl-9 pr-3 py-1.5 text-xs text-[#cdd6f4] placeholder-[#585b70] focus:outline-none focus:border-[#89b4fa] transition-colors"
                />
              </div>
            </div>
            <div className="flex-1 overflow-hidden">
              <VirtualGrid totalCount={totalCount} searchQuery={searchQuery} />
            </div>
          </div>
        )}

        {/* TAB 3: AI TRANSLATION & ROUTING */}
        {activeTab === 'ai' && (
          <div className="flex-1 overflow-y-auto p-8 max-w-4xl mx-auto w-full flex flex-col justify-center gap-6">
            <div className="bg-[#181825] border border-[#313244] rounded-2xl p-6 shadow-xl flex flex-col gap-5">
              <div className="flex items-center gap-3 border-b border-[#313244] pb-4">
                <Sparkles className="w-6 h-6 text-[#cba6f7]" />
                <div>
                  <h2 className="text-base font-bold text-[#cdd6f4]">Bước 3: Cấu Hình AI & Dịch Thuật Tự Động</h2>
                  <p className="text-xs text-[#a6adc8]">Định tuyến 3 bậc (TM/Glossary → 1-Pass LLM → TEaR + MBR) với bộ lập lịch DAG chống sập nguồn.</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-semibold text-[#bac2de]">Động cơ suy luận (Inference Engine)</label>
                  <select
                    value={aiEngine}
                    onChange={(e) => setAiEngine(e.target.value)}
                    className="w-full mt-1.5 bg-[#11111b] border border-[#313244] rounded-xl px-4 py-2.5 text-xs text-[#cdd6f4] focus:outline-none focus:border-[#89b4fa]"
                  >
                    <option value="llama.cpp">Local llama.cpp (GGUF Models, Tiết kiệm VRAM)</option>
                    <option value="vLLM">vLLM V1 (Model Runner V2, GPU Hiệu Năng Cao)</option>
                    <option value="SGLang">SGLang (Gia cố HMAC CVE-2026-7304)</option>
                    <option value="mock">Fast Simulation (Kiểm thử nhanh)</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs font-semibold text-[#bac2de]">Bảo toàn biến & Thẻ nội suy</label>
                  <div className="flex items-center gap-2 mt-2 px-4 py-2 bg-[#11111b] border border-[#313244] rounded-xl text-xs text-[#a6e3a1]">
                    <CheckCircle2 className="w-4 h-4" />
                    <span>renpy.substitutions 7 bước & ruby text tự động kích hoạt</span>
                  </div>
                </div>
              </div>

              <button
                onClick={handleStartAi}
                disabled={isLoading}
                className="flex items-center justify-center gap-2 px-5 py-3 bg-gradient-to-r from-[#cba6f7] to-[#89b4fa] hover:opacity-90 text-[#11111b] text-xs font-bold rounded-xl transition-all shadow-lg shadow-[#cba6f7]/20 disabled:opacity-50"
              >
                <Sparkles className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                Bắt Đầu Dịch Tự Động
              </button>

              {translatedProgress > 0 && (
                <div className="flex items-center justify-between p-4 bg-[#11111b] border border-[#a6e3a1]/30 rounded-xl">
                  <span className="text-xs text-[#cdd6f4]">Đã dịch thành công câu thoại:</span>
                  <span className="text-sm font-bold text-[#a6e3a1]">{translatedProgress} câu thoại</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 4: FONT AUDIT & OVERFLOW */}
        {activeTab === 'font' && (
          <div className="flex-1 overflow-y-auto p-8 max-w-4xl mx-auto w-full flex flex-col gap-6">
            <div className="bg-[#181825] border border-[#313244] rounded-2xl p-6 shadow-xl flex flex-col gap-5">
              <div className="flex items-center justify-between border-b border-[#313244] pb-4">
                <div className="flex items-center gap-3">
                  <Type className="w-6 h-6 text-[#f9e2af]" />
                  <div>
                    <h2 className="text-base font-bold text-[#cdd6f4]">Bước 4: Kiểm Toán Font 134 Ký Tự Tiếng Việt NFC</h2>
                    <p className="text-xs text-[#a6adc8]">Quét bảng cmap phát hiện ký tự thiếu và đo đạc tràn khung hộp thoại (HarfBuzz).</p>
                  </div>
                </div>
                <button
                  onClick={handleAuditFonts}
                  disabled={isLoading}
                  className="flex items-center gap-2 px-4 py-2 bg-[#f9e2af] hover:bg-[#f9e2af]/80 text-[#11111b] text-xs font-bold rounded-xl transition-all disabled:opacity-50"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
                  Kiểm Toán Font
                </button>
              </div>

              {fontAuditList.length > 0 && (
                <div className="flex flex-col gap-3">
                  <h3 className="text-xs font-semibold text-[#bac2de]">Danh sách Font trong dự án:</h3>
                  <div className="flex flex-col gap-2 max-h-48 overflow-y-auto pr-2">
                    {fontAuditList.map((f, idx) => (
                      <div
                        key={idx}
                        className={`flex items-center justify-between p-3 rounded-xl border text-xs ${
                          f.is_fully_compatible
                            ? 'bg-[#11111b] border-[#a6e3a1]/30 text-[#a6e3a1]'
                            : 'bg-[#11111b] border-[#f38ba8]/30 text-[#f38ba8]'
                        }`}
                      >
                        <span className="font-medium">{f.font_name}</span>
                        <span>{f.is_fully_compatible ? 'Tương thích 100% (134/134)' : `Thiếu ${f.missing_chars.length} glyphs`}</span>
                      </div>
                    ))}
                  </div>

                  {!isAllFontsCompatible && (
                    <div className="flex flex-col gap-2 mt-2">
                      <span className="text-xs font-semibold text-[#f9e2af] flex items-center gap-1.5">
                        <AlertTriangle className="w-4 h-4" />
                        Tự động sinh cấu hình FontGroup fallback DSL cho Ren'Py:
                      </span>
                      <pre className="bg-[#11111b] border border-[#313244] p-3 rounded-xl text-[11px] font-mono text-[#a6adc8] overflow-x-auto">
                        {fontGroupDsl}
                      </pre>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 5: EXPORT & LAUNCH */}
        {activeTab === 'export' && (
          <div className="flex-1 overflow-y-auto p-8 max-w-4xl mx-auto w-full flex flex-col justify-center gap-6">
            <div className="bg-[#181825] border border-[#313244] rounded-2xl p-6 shadow-xl flex flex-col gap-5">
              <div className="flex items-center gap-3 border-b border-[#313244] pb-4">
                <FileCheck className="w-6 h-6 text-[#a6e3a1]" />
                <div>
                  <h2 className="text-base font-bold text-[#cdd6f4]">Bước 5: Xuất Gói Dịch & Chơi Thử</h2>
                  <p className="text-xs text-[#a6adc8]">Tạo gói game/tl/vietnamese/ chuẩn 4 khoảng trắng, 0 tabs kèm Runtime Hook.</p>
                </div>
              </div>

              <div className="flex gap-4">
                <button
                  onClick={handleExport}
                  disabled={isLoading}
                  className="flex-1 flex items-center justify-center gap-2 px-6 py-3.5 bg-[#a6e3a1] hover:bg-[#a6e3a1]/90 text-[#11111b] text-xs font-bold rounded-xl transition-all shadow-lg shadow-[#a6e3a1]/20 disabled:opacity-50"
                >
                  <FileCheck className="w-4 h-4" />
                  Xuất Gói Dịch Chuẩn Ren'Py (1-Click)
                </button>
                <button
                  onClick={handleLaunchGame}
                  className="flex-1 flex items-center justify-center gap-2 px-6 py-3.5 bg-[#89b4fa] hover:bg-[#b4befe] text-[#11111b] text-xs font-bold rounded-xl transition-all shadow-lg shadow-[#89b4fa]/20"
                >
                  <Play className="w-4 h-4 fill-current" />
                  Khởi Chạy Game Ngay
                </button>
              </div>

              {exportedFiles.length > 0 && (
                <div className="flex flex-col gap-2 p-4 bg-[#11111b] border border-[#a6e3a1]/30 rounded-xl text-xs">
                  <span className="font-semibold text-[#a6e3a1]">Đã xuất thành công các tệp dịch vào {exportOutputDir}:</span>
                  <div className="flex flex-wrap gap-2 mt-1">
                    {exportedFiles.map((f, i) => (
                      <span key={i} className="px-2.5 py-1 rounded-md bg-[#181825] border border-[#313244] text-[#bac2de]">
                        {f}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Bottom Status Notification Bar */}
      <footer className="flex items-center justify-between px-6 py-2 bg-[#11111b] border-t border-[#313244] text-[11px] text-[#6c7086]">
        <div className="flex items-center gap-2 truncate max-w-2xl">
          <span className="w-1.5 h-1.5 rounded-full bg-[#89b4fa]" />
          <span className="truncate text-[#bac2de]">{statusMessage || 'Hệ thống sẵn sàng.'}</span>
        </div>
        <div className="flex items-center gap-4 text-[10px] text-[#585b70]">
          <span>PRAGMA WAL Mode</span>
          <span>•</span>
          <span>Zero-Copy VFS</span>
          <span>•</span>
          <span>90-Frame Storm Breaker</span>
        </div>
      </footer>
    </div>
  );
};

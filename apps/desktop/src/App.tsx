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
  getSystemSpecs,
  assessModel,
  getTranslationProgress,
  unloadModel,
  testInferenceConnection,
  FontAuditItem,
  SystemSpecs,
  ModelAssessment,
  TranslationProgress,
  ConnectionTestResult,
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
  Cpu,
  HardDrive,
  Zap,
  Trash2,
  AlertCircle,
  Database,
} from 'lucide-react';

type Tab = 'project' | 'editor' | 'ai' | 'font' | 'export';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<Tab>('project');
  const [gameDir, setGameDir] = useState<string>('');
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

  // Translation & Hardware state
  const [aiEngine, setAiEngine] = useState<string>('llama.cpp');
  const [translatedProgress, setTranslatedProgress] = useState<number>(0);
  const [systemSpecs, setSystemSpecs] = useState<SystemSpecs | null>(null);
  const [modelAssessment, setModelAssessment] = useState<ModelAssessment | null>(null);
  const [modelPath, setModelPath] = useState<string>('');
  const [serverUrl, setServerUrl] = useState<string>('');
  const [nGpuLayers, setNGpuLayers] = useState<number>(-1);
  const [nCtx, setNCtx] = useState<number>(2048);
  const [temperature, setTemperature] = useState<number>(0.2);
  const [batchLimit, setBatchLimit] = useState<number>(20);
  const [liveProgress, setLiveProgress] = useState<TranslationProgress | null>(null);
  const [translationError, setTranslationError] = useState<string>('');
  const [translationSuccess, setTranslationSuccess] = useState<string>('');
  const [connectionTest, setConnectionTest] = useState<ConnectionTestResult | null>(null);
  const [isTestingConnection, setIsTestingConnection] = useState<boolean>(false);

  // Export state
  const [exportedFiles, setExportedFiles] = useState<string[]>([]);
  const [exportOutputDir, setExportOutputDir] = useState<string>('');

  const fetchSpecs = async () => {
    try {
      const specs = await getSystemSpecs();
      setSystemSpecs(specs);
    } catch (e) {
      console.warn('Failed to load system specs:', e);
    }
  };

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
    fetchSpecs();
  }, []);

  // Assess memory risk whenever modelPath changes
  useEffect(() => {
    if (modelPath && modelPath.endsWith('.gguf')) {
      assessModel(modelPath)
        .then((res) => {
          setModelAssessment(res);
          if (res.recommended_ctx && nCtx > res.recommended_ctx) {
            setNCtx(res.recommended_ctx);
          }
        })
        .catch((e) => console.warn('Model assessment failed:', e));
    } else {
      setModelAssessment(null);
    }
  }, [modelPath]);

  // Poll progress during active translation
  useEffect(() => {
    let interval: any = null;
    if (isLoading && activeTab === 'ai') {
      interval = setInterval(async () => {
        try {
          const prog = await getTranslationProgress();
          setLiveProgress(prog);
        } catch (e) {
          console.warn(e);
        }
      }, 500);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isLoading, activeTab]);

  // 1. Scan Project
  const handleScanProject = async () => {
    if (!gameDir.trim()) return;
    setIsLoading(true);
    setStatusMessage('Đang phân tích cấu trúc game, gắn VFS đọc .rpa và trích xuất AST...');
    setTranslationError('');
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

  // 1b. Open Existing Project Database (.sqlite)
  const handleOpenExistingDb = async (customPath?: string) => {
    const pathToOpen = (customPath || dbPath).trim();
    if (!pathToOpen) return;
    setIsLoading(true);
    setStatusMessage('Đang kết nối vào cơ sở dữ liệu SQLite...');
    setTranslationError('');
    try {
      await openProjectDb(pathToOpen);
      setDbPath(pathToOpen);
      const count = await getTotalCount();
      setTotalCount(count);
      setStatusMessage(`Đã nạp cơ sở dữ liệu thành công! Tìm thấy ${count.toLocaleString()} câu thoại.`);
    } catch (err: any) {
      const msg = `Lỗi mở database: ${err.message || err}`;
      setStatusMessage(msg);
      setTranslationError(msg);
    } finally {
      setIsLoading(false);
    }
  };

  // 1c. Test llama-server / external inference connection
  const handleTestConnection = async () => {
    const targetUrl = serverUrl.trim() || 'http://127.0.0.1:8080/v1';
    setIsTestingConnection(true);
    setConnectionTest(null);
    try {
      const res = await testInferenceConnection(targetUrl);
      setConnectionTest(res);
      if (res.online) {
        setStatusMessage(`Kết nối llama-server thành công! ${res.models && res.models.length > 0 ? `(Model: ${res.models[0]})` : ''}`);
      } else {
        setStatusMessage(`Không thể kết nối llama-server: ${res.message}`);
      }
    } catch (err: any) {
      setConnectionTest({
        online: false,
        message: `Lỗi kết nối: ${err.message || err}`,
      });
    } finally {
      setIsTestingConnection(false);
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
    setTranslationError('');
    setTranslationSuccess('');

    if (!dbPath.trim()) {
      const msg = 'Vui lòng mở hoặc quét dự án trước khi dịch.';
      setStatusMessage(msg);
      setTranslationError(msg);
      return;
    }
    if (aiEngine === 'llama.cpp' && !modelPath.trim() && !serverUrl.trim()) {
      const msg = 'Vui lòng nhập đường dẫn file GGUF model (.gguf) hoặc URL máy chủ suy luận (server_url).';
      setStatusMessage(msg);
      setTranslationError(msg);
      return;
    }
    setIsLoading(true);
    setStatusMessage(`Đang chạy dịch tự động qua động cơ ${aiEngine}...`);
    try {
      const res = await startTranslation({
        db_path: dbPath.trim(),
        engine: aiEngine,
        model_path: modelPath.trim() || undefined,
        server_url: serverUrl.trim() || undefined,
        n_gpu_layers: nGpuLayers,
        n_ctx: nCtx,
        temperature: temperature,
        limit: batchLimit,
      });
      setTranslatedProgress(res.translated_count);
      const count = await getTotalCount();
      setTotalCount(count);
      if (res.translated_count === 0) {
        setTranslationSuccess('Đã hoàn tất kiểm tra: Toàn bộ câu thoại đã được dịch hoặc không có câu nào ở trạng thái chờ dịch (PENDING).');
        setStatusMessage('Không có câu thoại nào cần dịch thêm.');
      } else {
        setTranslationSuccess(`Đã hoàn tất dịch lô ${res.translated_count} câu thoại vào cơ sở dữ liệu!`);
        setStatusMessage(`Đã hoàn tất dịch lô ${res.translated_count} câu thoại!`);
      }
      await fetchSpecs();
    } catch (err: any) {
      const msg = `Lỗi dịch: ${err.message || err}`;
      setStatusMessage(msg);
      setTranslationError(msg);
    } finally {
      setIsLoading(false);
      try {
        const prog = await getTranslationProgress();
        setLiveProgress(prog);
      } catch (_) {}
    }
  };

  const handleUnloadModel = async () => {
    try {
      await unloadModel();
      setStatusMessage('Đã giải phóng model khỏi VRAM/RAM thành công.');
      await fetchSpecs();
    } catch (err: any) {
      setStatusMessage(`Lỗi khi giải phóng model: ${err.message || err}`);
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

              {/* Option to Open Existing Project Database */}
              <div className="flex flex-col gap-2 pt-4 border-t border-[#313244]">
                <div className="flex items-center gap-2">
                  <Database className="w-4 h-4 text-[#89b4fa]" />
                  <label className="text-xs font-semibold text-[#bac2de]">Hoặc Mở Tệp Cơ Sở Dữ Liệu SQLite Đã Có (.sqlite)</label>
                </div>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={dbPath}
                    onChange={(e) => setDbPath(e.target.value)}
                    placeholder="/duong/dan/toi/renslate_project.sqlite..."
                    className="flex-1 bg-[#11111b] border border-[#313244] rounded-xl px-4 py-2.5 text-xs text-[#cdd6f4] font-mono focus:outline-none focus:border-[#89b4fa] transition-colors"
                  />
                  <button
                    onClick={() => handleOpenExistingDb()}
                    disabled={isLoading || !dbPath.trim()}
                    className="flex items-center gap-2 px-5 py-2.5 bg-[#313244] hover:bg-[#45475a] text-[#89b4fa] text-xs font-bold rounded-xl transition-all disabled:opacity-50"
                  >
                    <Database className="w-4 h-4" />
                    Mở Database
                  </button>
                </div>
              </div>
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
              <div className="flex items-center justify-between border-b border-[#313244] pb-4">
                <div className="flex items-center gap-3">
                  <Sparkles className="w-6 h-6 text-[#cba6f7]" />
                  <div>
                    <h2 className="text-base font-bold text-[#cdd6f4]">Bước 3: Cấu Hình AI & Dịch Thuật Tự Động</h2>
                    <p className="text-xs text-[#a6adc8]">Hệ thống bảo vệ bộ nhớ _room_for_copy chống tràn OOM, hỗ trợ GGUF qua CUDA Offload.</p>
                  </div>
                </div>
                <button
                  onClick={handleUnloadModel}
                  disabled={isLoading}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-[#313244] hover:bg-[#45475a] text-[#f38ba8] text-xs font-semibold rounded-lg transition-all"
                  title="Giải phóng model khỏi VRAM/RAM"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Dọn VRAM
                </button>
              </div>

              {/* Project Database Status & Quick Loader in Tab 3 */}
              {dbPath ? (
                <div className="flex items-center justify-between p-3.5 bg-[#11111b] border border-[#313244] rounded-xl text-xs">
                  <div className="flex items-center gap-2.5 min-w-0">
                    <Database className="w-4 h-4 text-[#89b4fa] shrink-0" />
                    <span className="text-[#a6adc8] shrink-0">CSDL dự án:</span>
                    <span className="font-mono text-[#cdd6f4] truncate max-w-sm" title={dbPath}>{dbPath}</span>
                    <span className="px-2 py-0.5 rounded-full bg-[#89b4fa]/20 text-[#89b4fa] text-[10px] font-bold shrink-0">
                      {totalCount.toLocaleString()} câu thoại
                    </span>
                  </div>
                  <button
                    onClick={() => setActiveTab('project')}
                    className="text-[11px] text-[#89b4fa] hover:underline shrink-0"
                  >
                    Đổi CSDL
                  </button>
                </div>
              ) : (
                <div className="flex items-start gap-3 p-4 bg-[#fab387]/10 border border-[#fab387]/30 rounded-xl text-xs text-[#fab387]">
                  <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
                  <div className="flex-1 flex flex-col gap-2">
                    <div className="font-bold text-sm">Chưa Nạp Cơ Sở Dữ Liệu Dự Án</div>
                    <p className="text-[#cdd6f4]">
                      Vui lòng nhập đường dẫn tệp .sqlite hoặc quét thư mục game ở Tab 1 trước khi bắt đầu dịch.
                    </p>
                    <div className="flex gap-2 mt-1">
                      <input
                        type="text"
                        value={dbPath}
                        onChange={(e) => setDbPath(e.target.value)}
                        placeholder="/duong/dan/toi/renslate_project.sqlite..."
                        className="flex-1 bg-[#11111b] border border-[#313244] rounded-lg px-3 py-1.5 text-xs text-[#cdd6f4] font-mono focus:outline-none focus:border-[#89b4fa]"
                      />
                      <button
                        onClick={() => handleOpenExistingDb()}
                        disabled={!dbPath.trim() || isLoading}
                        className="px-4 py-1.5 bg-[#89b4fa] hover:bg-[#b4befe] text-[#11111b] font-bold rounded-lg transition-all disabled:opacity-50"
                      >
                        Nạp Database
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* Hardware Real-time Status Banner */}
              {systemSpecs && (
                <div className="grid grid-cols-3 gap-3 p-3 bg-[#11111b] border border-[#313244] rounded-xl text-xs">
                  <div className="flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-[#89b4fa]" />
                    <div>
                      <div className="font-semibold text-[#cdd6f4] truncate">{systemSpecs.gpu_name}</div>
                      <div className="text-[11px] text-[#a6adc8]">
                        VRAM trống: <span className="font-mono text-[#a6e3a1]">{(systemSpecs.free_vram_mb / 1024).toFixed(1)} GB</span> / {(systemSpecs.total_vram_mb / 1024).toFixed(1)} GB
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <HardDrive className="w-4 h-4 text-[#fab387]" />
                    <div>
                      <div className="font-semibold text-[#cdd6f4]">Bộ Nhớ Hệ Thống</div>
                      <div className="text-[11px] text-[#a6adc8]">
                        RAM trống: <span className="font-mono text-[#a6e3a1]">{(systemSpecs.free_ram_mb / 1024).toFixed(1)} GB</span> / {(systemSpecs.total_ram_mb / 1024).toFixed(1)} GB
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Zap className="w-4 h-4 text-[#f9e2af]" />
                    <div>
                      <div className="font-semibold text-[#cdd6f4]">Chống Tràn OOM</div>
                      <div className="text-[11px] text-[#a6adc8]">
                        {systemSpecs.swap_total_mb === 0 ? (
                          <span className="text-[#f9e2af] font-medium">Swap: 0B (Kích hoạt bảo vệ nghiêm ngặt)</span>
                        ) : (
                          <span>Swap: {(systemSpecs.swap_free_mb / 1024).toFixed(1)} GB</span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Engine Selector */}
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

              {/* GGUF Model Configuration */}
              {aiEngine === 'llama.cpp' && (
                <div className="flex flex-col gap-3 p-4 bg-[#11111b] border border-[#313244] rounded-xl">
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-semibold text-[#bac2de]">Đường dẫn tệp Model GGUF (.gguf)</label>
                    <input
                      type="text"
                      value={modelPath}
                      onChange={(e) => setModelPath(e.target.value)}
                      placeholder="/duong/dan/toi/model.gguf"
                      className="w-full bg-[#181825] border border-[#313244] rounded-xl px-4 py-2.5 text-xs text-[#cdd6f4] font-mono focus:outline-none focus:border-[#89b4fa]"
                    />
                  </div>

                  {/* Detected models suggestions */}
                  {systemSpecs?.detected_models && systemSpecs.detected_models.length > 0 && (
                    <div className="flex flex-col gap-1.5">
                      <span className="text-[11px] text-[#a6adc8]">Tệp model phát hiện được trong hệ thống:</span>
                      <div className="flex flex-wrap gap-1.5">
                        {systemSpecs.detected_models.map((m, idx) => (
                          <button
                            key={idx}
                            type="button"
                            onClick={() => setModelPath(m.path)}
                            className={`px-2.5 py-1 rounded-lg text-[11px] font-mono border transition-all ${
                              modelPath === m.path
                                ? 'bg-[#89b4fa]/20 border-[#89b4fa] text-[#89b4fa]'
                                : 'bg-[#181825] border-[#313244] text-[#a6adc8] hover:text-[#cdd6f4]'
                            }`}
                          >
                            {m.name} ({(m.size_mb / 1024).toFixed(1)} GB)
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* External llama-server URL alternative */}
                  <div className="flex flex-col gap-2 pt-2 border-t border-[#313244]">
                    <div className="flex items-center justify-between">
                      <label className="text-xs font-semibold text-[#bac2de]">
                        Hoặc kết nối máy chủ llama-server bên ngoài (OpenAI API - Khuyên Dùng Cho Model Lớn)
                      </label>
                      <button
                        type="button"
                        onClick={() => {
                          setServerUrl('http://127.0.0.1:8080/v1');
                          setConnectionTest(null);
                        }}
                        className="text-[11px] text-[#89b4fa] hover:underline font-mono"
                      >
                        ⚡ Điền nhanh: http://127.0.0.1:8080/v1
                      </button>
                    </div>
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={serverUrl}
                        onChange={(e) => {
                          setServerUrl(e.target.value);
                          setConnectionTest(null);
                        }}
                        placeholder="http://127.0.0.1:8080/v1 (Để trống nếu chạy tệp GGUF trực tiếp)"
                        className="flex-1 bg-[#181825] border border-[#313244] rounded-xl px-4 py-2 text-xs text-[#cdd6f4] font-mono focus:outline-none focus:border-[#89b4fa]"
                      />
                      <button
                        type="button"
                        onClick={handleTestConnection}
                        disabled={isTestingConnection}
                        className="flex items-center gap-1.5 px-3.5 py-2 bg-[#313244] hover:bg-[#45475a] text-[#cdd6f4] text-xs font-semibold rounded-xl transition-all disabled:opacity-50 shrink-0"
                      >
                        <RefreshCw className={`w-3.5 h-3.5 ${isTestingConnection ? 'animate-spin' : ''}`} />
                        Kiểm tra kết nối
                      </button>
                    </div>

                    {/* Connection Test Status */}
                    {connectionTest && (
                      <div
                        className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs ${
                          connectionTest.online
                            ? 'bg-[#a6e3a1]/10 text-[#a6e3a1] border border-[#a6e3a1]/30'
                            : 'bg-[#f38ba8]/10 text-[#f38ba8] border border-[#f38ba8]/30'
                        }`}
                      >
                        {connectionTest.online ? (
                          <CheckCircle2 className="w-4 h-4 shrink-0" />
                        ) : (
                          <AlertCircle className="w-4 h-4 shrink-0" />
                        )}
                        <span className="truncate">{connectionTest.message}</span>
                        {connectionTest.models && connectionTest.models.length > 0 && (
                          <span className="ml-auto font-mono text-[10px] bg-[#11111b] px-2 py-0.5 rounded">
                            {connectionTest.models[0]}
                          </span>
                        )}
                      </div>
                    )}
                  </div>

                  {/* OOM Assessment Card */}
                  {modelAssessment && (
                    <div
                      className={`flex items-start gap-2.5 p-3 rounded-xl border text-xs ${
                        modelAssessment.status === 'oom_risk'
                          ? 'bg-[#f38ba8]/10 border-[#f38ba8]/30 text-[#f38ba8]'
                          : modelAssessment.status === 'safe_full_gpu'
                          ? 'bg-[#a6e3a1]/10 border-[#a6e3a1]/30 text-[#a6e3a1]'
                          : 'bg-[#89b4fa]/10 border-[#89b4fa]/30 text-[#89b4fa]'
                      }`}
                    >
                      <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                      <div>
                        <div className="font-semibold">Đánh Giá Bộ Nhớ Khả Dụng:</div>
                        <div>{modelAssessment.message}</div>
                      </div>
                    </div>
                  )}

                  {/* Advanced inference parameters */}
                  <div className="grid grid-cols-4 gap-3 pt-2 border-t border-[#313244]">
                    <div>
                      <label className="text-[11px] font-semibold text-[#bac2de]">GPU Offload (Layers)</label>
                      <input
                        type="number"
                        value={nGpuLayers}
                        onChange={(e) => setNGpuLayers(parseInt(e.target.value) || 0)}
                        className="w-full mt-1 bg-[#181825] border border-[#313244] rounded-lg px-2.5 py-1.5 text-xs text-[#cdd6f4]"
                        title="-1 để tự động tối ưu hóa cho VRAM"
                      />
                      <span className="text-[10px] text-[#6c7086]">-1: Tự động</span>
                    </div>
                    <div>
                      <label className="text-[11px] font-semibold text-[#bac2de]">Ngữ Cảnh (n_ctx)</label>
                      <select
                        value={nCtx}
                        onChange={(e) => setNCtx(parseInt(e.target.value))}
                        className="w-full mt-1 bg-[#181825] border border-[#313244] rounded-lg px-2.5 py-1.5 text-xs text-[#cdd6f4]"
                      >
                        <option value={1024}>1024 (Tiết kiệm RAM nhất)</option>
                        <option value={2048}>2048 (Khuyên dùng)</option>
                        <option value={4096}>4096 (Ngữ cảnh dài)</option>
                      </select>
                      <span className="text-[10px] text-[#6c7086]">Giảm để chống OOM</span>
                    </div>
                    <div>
                      <label className="text-[11px] font-semibold text-[#bac2de]">Nhiệt Độ (Temp)</label>
                      <input
                        type="number"
                        step="0.05"
                        value={temperature}
                        onChange={(e) => setTemperature(parseFloat(e.target.value) || 0.2)}
                        className="w-full mt-1 bg-[#181825] border border-[#313244] rounded-lg px-2.5 py-1.5 text-xs text-[#cdd6f4]"
                      />
                      <span className="text-[10px] text-[#6c7086]">0.2: Ổn định</span>
                    </div>
                    <div>
                      <label className="text-[11px] font-semibold text-[#bac2de]">Lô Dịch (Câu/Đợt)</label>
                      <input
                        type="number"
                        value={batchLimit}
                        onChange={(e) => setBatchLimit(parseInt(e.target.value) || 20)}
                        className="w-full mt-1 bg-[#181825] border border-[#313244] rounded-lg px-2.5 py-1.5 text-xs text-[#cdd6f4]"
                      />
                      <span className="text-[10px] text-[#6c7086]">Giới hạn an toàn</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Inline Error Banner */}
              {translationError && (
                <div className="flex items-start gap-2.5 p-3.5 bg-[#f38ba8]/15 border border-[#f38ba8]/40 rounded-xl text-xs text-[#f38ba8]">
                  <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <span className="font-bold">Lỗi: </span>
                    <span>{translationError}</span>
                  </div>
                  <button
                    onClick={() => setTranslationError('')}
                    className="text-[#f38ba8] hover:text-[#cdd6f4] font-bold text-sm leading-none ml-2"
                  >
                    ✕
                  </button>
                </div>
              )}

              {/* Inline Success Banner */}
              {translationSuccess && (
                <div className="flex items-start gap-2.5 p-3.5 bg-[#a6e3a1]/15 border border-[#a6e3a1]/40 rounded-xl text-xs text-[#a6e3a1]">
                  <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <span className="font-bold">Thành công: </span>
                    <span>{translationSuccess}</span>
                  </div>
                  <button
                    onClick={() => setTranslationSuccess('')}
                    className="text-[#a6e3a1] hover:text-[#cdd6f4] font-bold text-sm leading-none ml-2"
                  >
                    ✕
                  </button>
                </div>
              )}

              {/* Start Button */}
              <button
                onClick={handleStartAi}
                disabled={isLoading}
                className="flex items-center justify-center gap-2 px-5 py-3 bg-gradient-to-r from-[#cba6f7] to-[#89b4fa] hover:opacity-90 text-[#11111b] text-xs font-bold rounded-xl transition-all shadow-lg shadow-[#cba6f7]/20 disabled:opacity-50"
              >
                <Sparkles className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                {isLoading ? 'Đang Thực Hiện Dịch...' : 'Bắt Đầu Dịch Tự Động'}
              </button>

              {/* Live Translation Progress Details */}
              {liveProgress && liveProgress.is_running && (
                <div className="flex flex-col gap-2.5 p-4 bg-[#11111b] border border-[#89b4fa]/30 rounded-xl text-xs">
                  <div className="flex justify-between items-center text-[#cdd6f4]">
                    <span className="font-semibold">Tiến độ đợt dịch:</span>
                    <span className="font-mono text-[#89b4fa]">
                      {liveProgress.current} / {liveProgress.total} câu (
                      {liveProgress.total > 0 ? Math.round((liveProgress.current / liveProgress.total) * 100) : 0}%)
                    </span>
                  </div>
                  <div className="w-full bg-[#181825] rounded-full h-1.5 overflow-hidden">
                    <div
                      className="bg-[#89b4fa] h-full transition-all duration-300"
                      style={{
                        width: `${
                          liveProgress.total > 0 ? (liveProgress.current / liveProgress.total) * 100 : 0
                        }%`,
                      }}
                    />
                  </div>
                  {liveProgress.current_source && (
                    <div className="flex flex-col gap-1 mt-1 font-mono text-[11px]">
                      <div className="text-[#a6adc8] truncate">Nguồn: {liveProgress.current_source}</div>
                      <div className="text-[#a6e3a1] truncate">Đang sinh: {liveProgress.current_target || '...'}</div>
                    </div>
                  )}
                </div>
              )}

              {translatedProgress > 0 && !isLoading && (
                <div className="flex items-center justify-between p-4 bg-[#11111b] border border-[#a6e3a1]/30 rounded-xl">
                  <span className="text-xs text-[#cdd6f4]">Đã dịch thành công tổng cộng:</span>
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

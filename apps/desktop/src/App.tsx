import React, { useState, useEffect } from 'react';
import { VirtualGrid } from './VirtualGrid';
import { getTotalCount, getSidecarStatus, openProjectDb } from './api';
import { Search, Database, Cpu, Layers, HardDrive, RefreshCw } from 'lucide-react';

export const App: React.FC = () => {
  const [totalCount, setTotalCount] = useState<number>(300_000);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [dbPath, setDbPath] = useState<string>('/path/to/project.sqlite');
  const [isSidecarAlive, setIsSidecarAlive] = useState<boolean>(true);
  const [stats, setStats] = useState({
    renderedNodes: 0,
    activeChunks: 0,
    latencyMs: 0,
  });

  useEffect(() => {
    // Initial fetch of total count and sidecar status
    const init = async () => {
      try {
        const count = await getTotalCount();
        setTotalCount(count);
        const alive = await getSidecarStatus();
        setIsSidecarAlive(alive);
      } catch (err) {
        console.warn('Initial connection notice:', err);
      }
    };
    init();
  }, []);

  const handleOpenDb = async () => {
    if (!dbPath.trim()) return;
    try {
      await openProjectDb(dbPath);
      const count = await getTotalCount();
      setTotalCount(count);
    } catch (err) {
      console.error('Failed to open database:', err);
    }
  };

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-[#181825] text-[#cdd6f4] font-sans">
      {/* Top Navigation Bar */}
      <header className="flex items-center justify-between px-5 py-3 bg-[#11111b] border-b border-[#313244] shadow-md z-10">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-gradient-to-br from-[#89b4fa] to-[#cba6f7] text-[#11111b] font-bold text-lg shadow-sm">
            R
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight text-[#cdd6f4]">
              Renslate <span className="text-xs font-normal text-[#89b4fa]">v0.1.0 (Tauri v2)</span>
            </h1>
            <p className="text-[11px] text-[#6c7086]">Local-First Ren'Py Localization Engine</p>
          </div>
        </div>

        {/* Database Path Input */}
        <div className="flex items-center gap-2 max-w-md w-full mx-4">
          <div className="relative flex-1">
            <Database className="absolute left-3 top-2.5 w-4 h-4 text-[#6c7086]" />
            <input
              type="text"
              value={dbPath}
              onChange={(e) => setDbPath(e.target.value)}
              placeholder="Path to project.sqlite..."
              className="w-full bg-[#181825] border border-[#313244] rounded-lg pl-9 pr-3 py-1.5 text-xs text-[#cdd6f4] placeholder-[#585b70] focus:outline-none focus:border-[#89b4fa] transition-colors"
            />
          </div>
          <button
            onClick={handleOpenDb}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-[#313244] hover:bg-[#45475a] text-xs font-medium text-[#cdd6f4] rounded-lg transition-colors border border-[#45475a]"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Connect
          </button>
        </div>

        {/* Search Bar */}
        <div className="relative w-80">
          <Search className="absolute left-3 top-2.5 w-4 h-4 text-[#6c7086]" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search FTS5 (source, target, speaker)..."
            className="w-full bg-[#181825] border border-[#313244] rounded-lg pl-9 pr-3 py-1.5 text-xs text-[#cdd6f4] placeholder-[#585b70] focus:outline-none focus:border-[#89b4fa] focus:ring-1 focus:ring-[#89b4fa] transition-colors"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-2.5 top-2 text-xs text-[#6c7086] hover:text-[#cdd6f4]"
            >
              ✕
            </button>
          )}
        </div>

        {/* Sidecar Status Indicator */}
        <div className="flex items-center gap-2 pl-2">
          <div
            className={`w-2.5 h-2.5 rounded-full ${
              isSidecarAlive ? 'bg-[#a6e3a1] shadow-[0_0_8px_rgba(166,227,161,0.6)]' : 'bg-[#f38ba8]'
            }`}
          />
          <span className="text-xs text-[#a6adc8] font-mono">
            {isSidecarAlive ? 'IPC Sidecar Active' : 'Sidecar Disconnected'}
          </span>
        </div>
      </header>

      {/* Main Grid View */}
      <main className="flex-1 overflow-hidden relative">
        <VirtualGrid
          totalCount={totalCount}
          searchQuery={searchQuery}
          onStatsUpdate={setStats}
        />
      </main>

      {/* Footer / Telemetry Bar */}
      <footer className="flex items-center justify-between px-5 py-2 bg-[#11111b] border-t border-[#313244] text-[11px] text-[#a6adc8] select-none">
        <div className="flex items-center gap-5">
          <div className="flex items-center gap-1.5">
            <HardDrive className="w-3.5 h-3.5 text-[#89b4fa]" />
            <span>Total Units: <strong className="text-[#cdd6f4] font-mono">{totalCount.toLocaleString()}</strong></span>
          </div>
          <div className="flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-[#f9e2af]" />
            <span>Active DOM Nodes: <strong className="text-[#cdd6f4] font-mono">{stats.renderedNodes}</strong></span>
          </div>
          <div className="flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-[#a6e3a1]" />
            <span>Chunks in Memory: <strong className="text-[#cdd6f4] font-mono">{stats.activeChunks} / 25</strong></span>
          </div>
        </div>

        <div className="flex items-center gap-4 font-mono">
          <span>Latency: <strong className="text-[#89b4fa]">{stats.latencyMs}ms</strong></span>
          <span className="px-2 py-0.5 rounded bg-[#313244]/80 text-[#a6e3a1] border border-[#45475a]">
            Memory: &lt; 80MB (WebKitGTK/WebView2)
          </span>
        </div>
      </footer>
    </div>
  );
};

import React, { useRef, useState, useEffect, useCallback } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { TranslationUnit } from './types';
import { fetchTranslationPage, searchUnits } from './api';

const PAGE_SIZE = 200;
const ROW_HEIGHT = 56;
const MAX_CACHED_CHUNKS = 25; // Keeps at most 5,000 items in memory (~3MB RAM)

interface VirtualGridProps {
  totalCount: number;
  searchQuery: string;
  onStatsUpdate?: (stats: {
    renderedNodes: number;
    activeChunks: number;
    latencyMs: number;
  }) => void;
}

export const VirtualGrid: React.FC<VirtualGridProps> = ({
  totalCount,
  searchQuery,
  onStatsUpdate,
}) => {
  const parentRef = useRef<HTMLDivElement>(null);

  // Sparse chunk cache: chunkIndex (0, 1, 2, ...) -> Array of 200 rows
  const [chunkCache, setChunkCache] = useState<Map<number, TranslationUnit[]>>(new Map());
  const fetchingChunks = useRef<Set<number>>(new Set());

  // Search state
  const [searchResults, setSearchResults] = useState<TranslationUnit[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [lastQueryLatency, setLastQueryLatency] = useState(0);

  const isSearchActive = searchQuery.trim().length > 0;
  const effectiveCount = isSearchActive ? searchResults.length : totalCount;

  // Virtualizer setup
  const virtualizer = useVirtualizer({
    count: effectiveCount,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 8,
  });

  const virtualItems = virtualizer.getVirtualItems();

  // FTS5 Search Execution
  useEffect(() => {
    let isCancelled = false;
    const query = searchQuery.trim();

    if (!query) {
      setSearchResults([]);
      setIsSearching(false);
      return;
    }

    setIsSearching(true);
    const start = performance.now();

    const timer = setTimeout(async () => {
      try {
        const results = await searchUnits(query, PAGE_SIZE);
        if (!isCancelled) {
          const latency = Math.round(performance.now() - start);
          setSearchResults(results);
          setLastQueryLatency(latency);
          setIsSearching(false);
          // Scroll back to top on new search
          parentRef.current?.scrollTo({ top: 0 });
        }
      } catch (err) {
        if (!isCancelled) {
          console.error('FTS5 search error:', err);
          setIsSearching(false);
        }
      }
    }, 150); // 150ms debounce

    return () => {
      isCancelled = true;
      clearTimeout(timer);
    };
  }, [searchQuery]);

  // Keyset page fetcher for chunkIndex
  const fetchChunk = useCallback(async (chunkIndex: number) => {
    if (fetchingChunks.current.has(chunkIndex)) return;
    fetchingChunks.current.add(chunkIndex);

    const cursorRowOrder = chunkIndex * PAGE_SIZE;
    const t0 = performance.now();

    try {
      const page = await fetchTranslationPage(cursorRowOrder, PAGE_SIZE);
      const latency = Math.round(performance.now() - t0);
      setLastQueryLatency(latency);

      setChunkCache((prev) => {
        const next = new Map(prev);
        next.set(chunkIndex, page);

        // LRU / distance eviction to enforce < 80MB memory budget
        if (next.size > MAX_CACHED_CHUNKS) {
          let furthestChunk = -1;
          let maxDist = -1;
          for (const key of next.keys()) {
            const dist = Math.abs(key - chunkIndex);
            if (dist > maxDist) {
              maxDist = dist;
              furthestChunk = key;
            }
          }
          if (furthestChunk !== -1) {
            next.delete(furthestChunk);
          }
        }
        return next;
      });
    } catch (err) {
      console.error(`Failed to fetch chunk ${chunkIndex}:`, err);
    } finally {
      fetchingChunks.current.delete(chunkIndex);
    }
  }, []);

  // Monitor visible items and trigger keyset fetches
  useEffect(() => {
    if (isSearchActive || virtualItems.length === 0) return;

    const visibleChunkIndices = new Set<number>();
    for (const item of virtualItems) {
      const chunkIdx = Math.floor(item.index / PAGE_SIZE);
      visibleChunkIndices.add(chunkIdx);
    }

    for (const chunkIdx of visibleChunkIndices) {
      if (!chunkCache.has(chunkIdx)) {
        fetchChunk(chunkIdx);
      }
    }
  }, [virtualItems, chunkCache, isSearchActive, fetchChunk]);

  // Report telemetry / performance stats
  useEffect(() => {
    if (onStatsUpdate) {
      onStatsUpdate({
        renderedNodes: virtualItems.length,
        activeChunks: chunkCache.size,
        latencyMs: lastQueryLatency,
      });
    }
  }, [virtualItems.length, chunkCache.size, lastQueryLatency, onStatsUpdate]);

  // Helper to retrieve a single row by its absolute virtual index
  const getItemAt = useCallback(
    (index: number): TranslationUnit | null => {
      if (isSearchActive) {
        return searchResults[index] ?? null;
      }
      const chunkIndex = Math.floor(index / PAGE_SIZE);
      const offsetInChunk = index % PAGE_SIZE;
      const chunk = chunkCache.get(chunkIndex);
      return chunk ? chunk[offsetInChunk] ?? null : null;
    },
    [isSearchActive, searchResults, chunkCache]
  );

  return (
    <div className="virtual-grid-container flex flex-col h-full w-full bg-[#1e1e2e] text-[#cdd6f4]">
      {/* Table Header */}
      <div className="grid grid-cols-[80px_140px_100px_1fr_1fr_110px] items-center px-4 py-3 bg-[#181825] border-b border-[#313244] font-semibold text-xs tracking-wider text-[#a6adc8] uppercase select-none">
        <div># Order</div>
        <div>Identifier</div>
        <div>Speaker</div>
        <div>Source (Ren'Py)</div>
        <div>Target (Vietnamese)</div>
        <div className="text-right">Status</div>
      </div>

      {/* Virtualized Scroll Viewport */}
      <div
        ref={parentRef}
        className="virtual-scroll-viewport flex-1 overflow-y-auto relative outline-none"
        style={{ contain: 'strict' }}
      >
        {effectiveCount === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-[#6c7086]">
            {isSearching ? (
              <div className="text-sm">Searching FTS5 virtual table...</div>
            ) : (
              <div className="text-sm">No translation units found</div>
            )}
          </div>
        ) : (
          <div
            style={{
              height: `${virtualizer.getTotalSize()}px`,
              width: '100%',
              position: 'relative',
            }}
          >
            {virtualItems.map((virtualRow) => {
              const item = getItemAt(virtualRow.index);
              const isEven = virtualRow.index % 2 === 0;

              return (
                <div
                  key={virtualRow.index}
                  className={`grid grid-cols-[80px_140px_100px_1fr_1fr_110px] items-center px-4 text-sm border-b border-[#313244]/40 transition-colors ${
                    isEven ? 'bg-[#1e1e2e]' : 'bg-[#181825]/60'
                  } hover:bg-[#313244]/50`}
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: `${virtualRow.size}px`,
                    transform: `translateY(${virtualRow.start}px)`,
                  }}
                >
                  {item ? (
                    <>
                      <div className="font-mono text-xs text-[#a6adc8]">
                        #{item.row_order}
                      </div>
                      <div className="font-mono text-xs text-[#89b4fa] truncate" title={item.unit_identifier}>
                        {item.unit_identifier}
                      </div>
                      <div className="text-xs text-[#fab387] font-medium truncate">
                        {item.speaker || <span className="text-[#585b70] italic">narrator</span>}
                      </div>
                      <div className="text-xs text-[#cdd6f4] pr-3 truncate" title={item.source_text}>
                        {item.source_text}
                      </div>
                      <div className="text-xs text-[#a6e3a1] pr-3 truncate" title={item.target_text || ''}>
                        {item.target_text ? (
                          item.target_text
                        ) : (
                          <span className="text-[#585b70] italic">Chưa dịch</span>
                        )}
                      </div>
                      <div className="text-right">
                        <span
                          className={`inline-block px-2 py-0.5 rounded text-[11px] font-semibold tracking-wide ${
                            item.status === 'COMMITTED'
                              ? 'bg-[#a6e3a1]/15 text-[#a6e3a1] border border-[#a6e3a1]/30'
                              : item.status === 'REVIEW'
                              ? 'bg-[#f9e2af]/15 text-[#f9e2af] border border-[#f9e2af]/30'
                              : item.status === 'DRAFT'
                              ? 'bg-[#89b4fa]/15 text-[#89b4fa] border border-[#89b4fa]/30'
                              : 'bg-[#6c7086]/20 text-[#a6adc8] border border-[#45475a]'
                          }`}
                        >
                          {item.status}
                        </span>
                      </div>
                    </>
                  ) : (
                    // Skeleton row while chunk is streaming from rusqlite ReadOnlyPool
                    <div className="col-span-6 flex items-center gap-3 animate-pulse">
                      <div className="h-4 w-12 bg-[#313244] rounded"></div>
                      <div className="h-4 w-24 bg-[#313244] rounded"></div>
                      <div className="h-4 w-16 bg-[#313244] rounded"></div>
                      <div className="h-4 flex-1 bg-[#313244] rounded"></div>
                      <div className="h-4 flex-1 bg-[#313244] rounded"></div>
                      <div className="h-4 w-16 bg-[#313244] rounded"></div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

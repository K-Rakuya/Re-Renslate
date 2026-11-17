# ==============================================================================
# Ren'Py In-Game VFS Fallback Loader for Renslate (ZiX-12A / ZiX-12B Archives)
# ==============================================================================
# Priority -999 ensures this initializes immediately after renpy.loader is set up,
# before game scripts, translation stores, or UI components execute.
# ==============================================================================

init -999 python:
    import os
    import sys
    import json
    import renpy

    class RenslateVfsFallback(object):
        """
        Safe In-Game VFS Fallback bridge for proprietary or encrypted archives
        (e.g., ZiX-12A, ZiX-12B) where static offline unpickling/extraction is not possible.
        Uses renpy.loader.load() within the live Ren'Py runtime environment.
        """

        @classmethod
        def can_load(cls, filename):
            """Checks whether renpy.loader can locate and load the given filename."""
            try:
                if hasattr(renpy.loader, "loadable"):
                    return renpy.loader.loadable(filename)
                return False
            except Exception:
                return False

        @classmethod
        def safe_load(cls, filename):
            """
            Safely loads raw binary content using renpy.loader.load().
            Returns bytes on success, or None on failure.
            """
            try:
                if not cls.can_load(filename):
                    return None
                with renpy.loader.load(filename) as f:
                    return f.read()
            except Exception as exc:
                try:
                    renpy.log("Renslate VFS Fallback load error: %s" % (exc,))
                except Exception:
                    pass
                return None

        @classmethod
        def list_all_files(cls):
            """
            Returns a sorted list of all virtual file paths known to renpy.loader.
            Traverses renpy.loader.listdirfiles() safely.
            """
            files = set()
            try:
                if hasattr(renpy.loader, "listdirfiles"):
                    for dirpath, file_list in renpy.loader.listdirfiles():
                        for fn in file_list:
                            normalized = fn.replace("\\", "/").lstrip("/")
                            files.add(normalized)
            except Exception as exc:
                try:
                    renpy.log("Renslate VFS Fallback listdirfiles error: %s" % (exc,))
                except Exception:
                    pass
            return sorted(list(files))

        @classmethod
        def dump_file(cls, filename, target_path):
            """Dumps a single virtual file to disk safely."""
            data = cls.safe_load(filename)
            if data is None:
                return False
            target_dir = os.path.dirname(target_path)
            if target_dir and not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
            with open(target_path, "wb") as f:
                f.write(data)
            return True

        @classmethod
        def dump_matching_files(cls, target_dir, extensions=(".rpy", ".rpyc", ".rpym", ".rpymc", ".json", ".txt")):
            """
            Extracts matching files to target_dir.
            Returns list of successfully dumped files.
            """
            all_files = cls.list_all_files()
            dumped = []
            for fn in all_files:
                if any(fn.lower().endswith(ext) for ext in extensions):
                    out_path = os.path.join(target_dir, fn)
                    if cls.dump_file(fn, out_path):
                        dumped.append(fn)
            return dumped

        @classmethod
        def export_manifest(cls, manifest_path):
            """Exports a JSON manifest of all loadable files in the VFS."""
            all_files = cls.list_all_files()
            manifest = {
                "version": "1.0",
                "total_files": len(all_files),
                "files": all_files,
            }
            manifest_dir = os.path.dirname(manifest_path)
            if manifest_dir and not os.path.exists(manifest_dir):
                os.makedirs(manifest_dir, exist_ok=True)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            return manifest

    try:
        _game_dir = getattr(renpy.config, "gamedir", None) or os.getcwd()
        _trigger_file = os.path.join(_game_dir, ".renslate_dump_trigger")
        _env_dump = os.environ.get("RENSLATE_FALLBACK_DUMP")
        _out_dir = os.environ.get("RENSLATE_FALLBACK_OUTPUT_DIR") or os.path.join(_game_dir, ".renslate_fallback_extracted")
        if _env_dump or os.path.exists(_trigger_file):
            RenslateVfsFallback.dump_matching_files(_out_dir)
            RenslateVfsFallback.export_manifest(os.path.join(_out_dir, "renslate_manifest.json"))
            if os.path.exists(_trigger_file):
                try:
                    os.remove(_trigger_file)
                except Exception:
                    pass
    except Exception:
        pass

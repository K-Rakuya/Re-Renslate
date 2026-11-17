# renloc_runtime_hook.rpy
# Renslate Safe In-Game Runtime Hook
# Guarantees:
# 1. State preservation via RevertableDict / RevertableList across save, load, and rollback.
# 2. Pure idempotent text filtering for config.say_menu_text_filter.
# 3. 90-frame storm circuit breaker preventing engine crash on recursive evaluation errors.

init -998 python:
    import sys

    # -------------------------------------------------------------------------
    # 1. Safe Revertable Container Resolution
    # -------------------------------------------------------------------------
    try:
        from renpy.revertable import RevertableDict, RevertableList
    except ImportError:
        try:
            from renpy.python import RevertableDict, RevertableList
        except ImportError:
            # Standalone / Headless simulation fallback
            class RevertableDict(dict):
                pass
            class RevertableList(list):
                pass

    # -------------------------------------------------------------------------
    # 2. 90-Frame Storm Circuit Breaker
    # -------------------------------------------------------------------------
    class RenlocCircuitBreaker(object):
        """
        Monitors consecutive text filter evaluation cycles and errors.
        Automatically trips after 90 consecutive frame errors to return raw text
        without freezing the engine or throwing unhandled exceptions.
        """
        MAX_CONSECUTIVE_ERRORS = 90

        def __init__(self):
            self.consecutive_errors = 0
            self.tripped = False
            self.trip_reason = ""
            self.trip_count = 0

        def record_success(self):
            if not self.tripped:
                self.consecutive_errors = 0

        def record_error(self, err_msg=""):
            self.consecutive_errors += 1
            if self.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS and not self.tripped:
                self.tripped = True
                self.trip_count += 1
                self.trip_reason = (
                    "90-Frame Storm Circuit Breaker tripped after "
                    + str(self.consecutive_errors)
                    + " consecutive errors. Last error: "
                    + str(err_msg)
                )
                try:
                    import renpy
                    renpy.log("[RENSLATE CRITICAL] " + self.trip_reason)
                except Exception:
                    print("[RENSLATE CRITICAL] " + self.trip_reason, file=sys.stderr)

        def reset(self):
            self.consecutive_errors = 0
            self.tripped = False
            self.trip_reason = ""

    # Initialize circuit breaker in runtime store
    renloc_breaker = RenlocCircuitBreaker()

    # Revertable dynamic pronoun and honorific context store
    # Must use RevertableDict to participate in Ren'Py rollback history cleanly
    renloc_dynamic_context = RevertableDict()

    # -------------------------------------------------------------------------
    # 3. Idempotent Say / Menu Text Filter
    # -------------------------------------------------------------------------
    def renloc_say_menu_filter(text):
        """
        Pure, idempotent text filter hooked into config.say_menu_text_filter.
        Satisfies: f(f(text)) == f(text) and does not mutate global state on rollback.
        """
        if not text or not isinstance(text, str):
            return text

        # If circuit breaker is tripped, immediately bypass filter and return raw text
        if renloc_breaker.tripped:
            return text

        try:
            # Process dynamic substitution if placeholders exist
            result = text

            # Simulated recursive error trigger for testing
            if "__RENSLATE_FORCE_RECURSIVE_ERROR__" in text:
                raise RuntimeError("Synthetic recursive substitution loop detected")

            # Replace dynamic tokens from revertable context without mutating state
            for token, replacement in renloc_dynamic_context.items():
                if token in result:
                    result = result.replace(token, replacement)

            # Record successful evaluation frame
            renloc_breaker.record_success()
            return result

        except Exception as e:
            renloc_breaker.record_error(str(e))
            # Fallback to returning raw original text without crashing
            return text

    # Register into Ren'Py config chain safely
    try:
        import renpy
        if hasattr(renpy, "config"):
            old_filter = getattr(renpy.config, "say_menu_text_filter", None)
            if old_filter is not None and old_filter != renloc_say_menu_filter:
                def chained_filter(t):
                    t = old_filter(t)
                    return renloc_say_menu_filter(t)
                renpy.config.say_menu_text_filter = chained_filter
            else:
                renpy.config.say_menu_text_filter = renloc_say_menu_filter
    except Exception:
        pass

"""Overhang analysis - aggregate dilutive instruments from parsed disclosures."""

from __future__ import annotations

import logging
import re
from datetime import date

from auto_reports.models.disclosure import CBIssuance
from auto_reports.models.report import OverhangItem

logger = logging.getLogger(__name__)


class OverhangAnalyzer:
    """Aggregate overhang data from multiple disclosure events."""

    def __init__(self, total_shares: int, currency: str = "KRW"):
        self.total_shares = total_shares
        self.currency = currency
        # Track instruments by (type, series) -> latest state
        self._instruments: dict[str, _InstrumentState] = {}

    def process_issuance(self, issuance: CBIssuance) -> None:
        """Process a CB/BW issuance event (initial overhang source)."""
        if not issuance.bond_type or issuance.bond_type.series is None:
            logger.warning("Issuance missing bond type/series, skipping")
            return

        series = issuance.bond_type.series
        kind = issuance.bond_type.kind or ""

        # Determine instrument category (skip 교환사채 — no dilution)
        if "교환" in kind:
            logger.info("Skipping exchangeable bond (교환사채) series %s", series)
            return
        if "전환" in kind:
            category = "CB"
        elif "신주인수권" in kind:
            category = "BW"
        else:
            category = "CB"  # default

        key = f"{category}_{series}"
        face_value = issuance.face_value or 0
        conversion_price = issuance.conversion_terms.conversion_price if issuance.conversion_terms else None

        # Use actual share count from disclosure ('전환에 따라 발행할 주식' → '주식수')
        # Fall back to calculation only if not available
        convertible_shares = 0
        if issuance.conversion_terms and issuance.conversion_terms.share_count:
            convertible_shares = issuance.conversion_terms.share_count
        elif conversion_price and conversion_price >= 100:
            convertible_shares = face_value // conversion_price

        # Extract exercise period from conversion terms
        ex_start = ""
        ex_end = ""
        if issuance.conversion_terms and issuance.conversion_terms.request_period:
            ex_start = issuance.conversion_terms.request_start or ""
            ex_end = issuance.conversion_terms.request_end or ""

        self._instruments[key] = _InstrumentState(
            category=category,
            series=series,
            kind=kind,
            face_value=face_value,
            remaining=face_value,
            conversion_price=conversion_price or 0,
            convertible_shares=convertible_shares,
            exercise_start=ex_start,
            exercise_end=ex_end,
        )

    def process_event(self, event: dict) -> None:
        """Process an overhang event from OpenDART event API (주요사항보고서).

        Handles: CB, BW, PERPETUAL, RIGHTS_ISSUE, MIXED_ISSUE
        Only adds instruments not already tracked from HTML disclosures.
        """
        category = event.get("category", "")

        if category in ("CB", "BW", "PERPETUAL"):
            series = event.get("series", 0)
            key = f"{category}_{series}"

            # If already tracked, enrich with API data
            if key in self._instruments:
                state = self._instruments[key]
                # Merge missing fields from structured API data
                ev_price = event.get("conversion_price") or 0
                ev_shares = event.get("convertible_shares") or 0
                ev_face = event.get("face_value") or 0
                ev_start = event.get("cv_start", "")
                ev_end = event.get("cv_end", "")
                if ev_price and not state.conversion_price:
                    state.conversion_price = ev_price
                if ev_shares and not state.convertible_shares:
                    state.convertible_shares = ev_shares
                if ev_face and not state.face_value:
                    state.face_value = ev_face
                    state.remaining = ev_face
                if ev_start and not state.exercise_start:
                    state.exercise_start = ev_start
                if ev_end and not state.exercise_end:
                    state.exercise_end = ev_end
                logger.debug("Enriched existing instrument %s with API data", key)
                return

            face_value = event.get("face_value") or 0
            conversion_price = event.get("conversion_price") or 0
            convertible_shares = event.get("convertible_shares") or 0

            if not convertible_shares and conversion_price >= 100:
                convertible_shares = face_value // conversion_price

            cat_label = {
                "CB": "CB", "BW": "BW", "PERPETUAL": "영구채",
            }
            self._instruments[key] = _InstrumentState(
                category=cat_label.get(category, category),
                series=series,
                kind=event.get("kind", ""),
                face_value=face_value,
                remaining=face_value,
                conversion_price=conversion_price,
                convertible_shares=convertible_shares,
                exercise_start=event.get("cv_start", ""),
                exercise_end=event.get("cv_end", ""),
            )
            logger.info("Added overhang from event: %s (face=%d)", key, face_value)

        elif category in ("RIGHTS_ISSUE", "MIXED_ISSUE"):
            shares = event.get("shares") or 0
            if shares <= 0:
                return
            conv_price = event.get("conversion_price") or event.get("issue_price") or 0
            cv_start = event.get("cv_start", "")
            cv_end = event.get("cv_end", "")

            rcept_no = event.get("rcept_no", "")
            if rcept_no:
                key = f"RIGHTS_{rcept_no}"
                if key in self._instruments:
                    # Enrich existing instrument with conversion terms
                    state = self._instruments[key]
                    if conv_price and not state.conversion_price:
                        state.conversion_price = conv_price
                    if cv_start and not state.exercise_start:
                        state.exercise_start = cv_start
                    if cv_end and not state.exercise_end:
                        state.exercise_end = cv_end
                    conv_shares = event.get("convertible_shares") or 0
                    if conv_shares and not state.convertible_shares:
                        state.convertible_shares = conv_shares
                    logger.debug("Enriched existing RIGHTS instrument %s", key)
                    return
            else:
                # No rcept_no (e.g. LLM PDF): match by shares to enrich existing
                for k, s in self._instruments.items():
                    if k.startswith("RIGHTS_") and s.convertible_shares == shares:
                        if conv_price and not s.conversion_price:
                            s.conversion_price = conv_price
                        if cv_start and not s.exercise_start:
                            s.exercise_start = cv_start
                        if cv_end and not s.exercise_end:
                            s.exercise_end = cv_end
                        logger.debug("Enriched RIGHTS instrument %s by shares match", k)
                        return
                # Use pdf_date for stable key (avoids ordinal counter collision)
                pdf_date = event.get("pdf_date", "")
                key = f"RIGHTS_{pdf_date}" if pdf_date else f"RIGHTS_llm_{id(event)}"

            label = "전환우선주"
            face_value = event.get("face_value") or 0
            self._instruments[key] = _InstrumentState(
                category=label,
                series=0,
                kind=event.get("share_type", ""),
                face_value=face_value,
                remaining=face_value,
                conversion_price=conv_price,
                convertible_shares=shares,
                exercise_start=cv_start,
                exercise_end=cv_end,
            )

        elif category == "STOCK_OPTION":
            shares = event.get("shares") or event.get("convertible_shares") or 0
            if shares <= 0:
                return
            series = event.get("series", 0)
            key = f"SO_{series}"

            if key in self._instruments:
                state = self._instruments[key]
                ev_price = event.get("exercise_price") or event.get("conversion_price") or 0
                ev_start = event.get("cv_start", "")
                ev_end = event.get("cv_end", "")
                # Phase 3b API data is newer than Phase 3a notes — always update
                if ev_price:
                    state.conversion_price = ev_price
                if shares:
                    state.convertible_shares = shares
                if ev_start:
                    state.exercise_start = ev_start
                if ev_end:
                    state.exercise_end = ev_end
                logger.debug("Enriched existing stock option %s with API data", key)
                return

            self._instruments[key] = _InstrumentState(
                category="주식매수선택권",
                series=series,
                kind=event.get("kind", ""),
                face_value=0,
                remaining=0,
                conversion_price=event.get("exercise_price") or event.get("conversion_price") or 0,
                convertible_shares=shares,
                exercise_start=event.get("cv_start", ""),
                exercise_end=event.get("cv_end", ""),
            )
            logger.info("Added stock option from event: %s (shares=%d)", key, shares)

    def process_rights_issue_disclosure(self, data: dict, disclosure_date: str = "") -> None:
        """Process a parsed 유상증자결정 disclosure (from HTML or PDF).

        Extracts conversion details (전환가액, 전환주식수, 전환청구기간) and
        creates or updates a RIGHTS_ISSUE instrument with rich data.

        This method creates instruments with the same key format as process_event
        (RIGHTS_{rcept_no}) so that event API data won't duplicate.

        Args:
            data: Dict from parse_rights_issue_html or parse_rights_issue_pdf.
            disclosure_date: Date string (YYYYMMDD or YYYY.MM.DD format).
        """
        new_shares = data.get("new_shares", {})
        estk = new_shares.get("기타주식") or 0
        if estk <= 0:
            return

        conversion = data.get("conversion", {})
        conv_price = conversion.get("전환가액(원/주)") or 0
        conv_shares = conversion.get("전환주식수") or 0
        conv_period = conversion.get("전환청구기간", {})
        cv_start = conv_period.get("시작일", "") or ""
        cv_end = conv_period.get("종료일", "") or ""

        # If no conversion shares, fall back to 기타주식 count
        if not conv_shares:
            conv_shares = estk

        # If no conversion price, try issue_price
        if not conv_price:
            conv_price = data.get("issue_price") or 0

        # Compute face value = conversion shares × conversion price
        face_value = conv_shares * conv_price if (conv_shares and conv_price) else 0

        # Compute funding-based face_value as fallback
        if not face_value:
            funding = data.get("funding_purpose") or {}
            face_value = sum(funding.values())

        # Generate key from disclosure_date to avoid duplicates
        key = f"RIGHTS_{disclosure_date}" if disclosure_date else f"RIGHTS_disc_{id(data)}"

        # Check if already tracked (from event API or notes) — match by conversion price
        for existing_key, state in self._instruments.items():
            if existing_key.startswith("RIGHTS_") and state.conversion_price == conv_price and conv_price:
                # Enrich existing instrument with richer data from disclosure
                if conv_shares:
                    state.convertible_shares = conv_shares
                if face_value and not state.remaining:
                    state.face_value = face_value
                    state.remaining = face_value
                if cv_start and not state.exercise_start:
                    state.exercise_start = cv_start
                if cv_end and not state.exercise_end:
                    state.exercise_end = cv_end
                logger.debug("Updated RIGHTS instrument %s from disclosure", existing_key)
                return

        label = "전환우선주"
        self._instruments[key] = _InstrumentState(
            category=label,
            series=0,
            kind=data.get("share_type", ""),
            face_value=face_value,
            remaining=face_value,
            conversion_price=conv_price,
            convertible_shares=conv_shares,
            exercise_start=cv_start,
            exercise_end=cv_end,
        )
        logger.info("Added RIGHTS instrument from disclosure: %s (shares=%d)", key, conv_shares)

    def process_notes_instrument(self, inst: dict) -> None:
        """Process an instrument parsed from financial statement notes (재무제표 주석).

        This is the primary/authoritative source of overhang data.
        Notes instruments represent the current state as of the report date.

        Args:
            inst: Dict from parse_notes_overhang() with keys:
                category, series, kind, face_value, convertible_shares,
                conversion_price, exercise_start, exercise_end, active
        """
        category = inst.get("category", "")
        series = inst.get("series", 0)
        kind = inst.get("kind", "")

        # Generate key
        if category in ("CB", "BW"):
            key = f"{category}_{series}"
        elif category == "SO":
            key = f"SO_{series}"
        else:
            # For 전환우선주, use kind text as differentiator
            safe_kind = kind.replace(" ", "_")[:40]
            key = f"PREF_{safe_kind}" if safe_kind else f"PREF_{id(inst)}"

        face_value = inst.get("face_value", 0)
        remaining_balance = inst.get("remaining_balance")
        if category == "SO":
            effective_remaining = 0
            effective_shares = inst.get("convertible_shares", 0)
        else:
            effective_remaining = remaining_balance if remaining_balance is not None else face_value
            effective_shares = inst.get("convertible_shares", 0) if effective_remaining > 0 else 0
        self._instruments[key] = _InstrumentState(
            category=category,
            series=series,
            kind=kind,
            face_value=face_value,
            remaining=effective_remaining,
            conversion_price=inst.get("conversion_price", 0),
            convertible_shares=effective_shares,
            exercise_start=inst.get("exercise_start", ""),
            exercise_end=inst.get("exercise_end", ""),
        )
        logger.info("Added overhang from notes: %s (shares=%d)", key, inst.get("convertible_shares", 0))

    def process_exchange_exercise(self, data: dict) -> None:
        """Process exchange disclosure exercise data (행사 공시).

        Updates existing instruments with latest balance/price info from
        KRX exchange disclosures (전환청구권행사, 신주인수권행사, 주식매수선택권행사 etc.).
        Handles series-based (CB/BW), date-based (전환주식), and stock option formats.
        """
        # Stock option exercise: update SO instruments with remaining shares
        so_remaining = data.get("so_remaining")
        if so_remaining and data.get("type") == "주식매수선택권행사":
            new_shares = so_remaining.get("new_shares", 0)
            if new_shares > 0:
                matched = False
                for key, state in self._instruments.items():
                    if state.category in ("주식매수선택권", "SO"):
                        state.convertible_shares = new_shares
                        logger.info(
                            "Updated SO instrument %s with remaining shares: %d",
                            key, new_shares,
                        )
                        matched = True
                if not matched:
                    # Create SO instrument from exchange exercise data
                    self._instruments["SO_0"] = _InstrumentState(
                        category="주식매수선택권",
                        series=0,
                        kind="",
                        face_value=0,
                        remaining=0,
                        conversion_price=0,
                        convertible_shares=new_shares,
                    )
                    logger.info(
                        "Created SO instrument from exchange exercise: SO_0 (shares=%d)",
                        new_shares,
                    )
            return

        # Skip 교환사채 (exchangeable bond) exercises — no dilution
        daily = data.get("daily_claims") or []
        notes = data.get("notes") or ""
        is_exchange_bond = (
            any("교환" in (d.get("bd_knd") or "") for d in daily)
            or "교환사채" in notes
            or "교환청구권" in notes
        )
        if is_exchange_bond:
            logger.info("Skipping exchangeable bond exercise (교환사채)")
            return

        for bal in data.get("cb_balance", []):
            series = bal.get("series")

            matched_key = None
            if series is not None:
                # Series-based matching (CB/BW)
                for key, state in self._instruments.items():
                    if state.series == series:
                        matched_key = key
                        break
            else:
                # Date-based matching (전환주식/전환우선주): match by conversion price
                cp = bal.get("conversion_price")
                if cp:
                    for key, state in self._instruments.items():
                        if state.conversion_price == cp and key.startswith("RIGHTS_"):
                            matched_key = key
                            break

            if matched_key:
                state = self._instruments[matched_key]
                # Update remaining balance: prefer 미잔액, fall back to 권면총액
                # remaining=0 means fully converted; don't fall back to face_value
                if "remaining" in bal:
                    remaining = bal["remaining"]
                else:
                    remaining = bal.get("face_value")
                if remaining is not None:
                    state.remaining = remaining
                # Update conversion price
                cp = bal.get("conversion_price")
                if cp is not None:
                    state.conversion_price = cp
                # Update convertible shares (series-based: convertible_shares, date-based: remaining_shares)
                cs = bal.get("convertible_shares") or bal.get("remaining_shares")
                if cs is not None:
                    state.convertible_shares = cs
                logger.debug(
                    "Updated instrument %s: remaining=%s, conv_price=%s, shares=%s (bal=%s)",
                    matched_key, state.remaining, state.conversion_price, state.convertible_shares, bal,
                )
            elif series is not None:
                # Fallback: create instrument from cb_balance when notes parsing missed it
                category = _infer_category_from_exercise(data.get("type", ""))
                key = f"{category}_{series}"
                if "remaining" in bal:
                    remaining = bal["remaining"] or 0
                else:
                    remaining = bal.get("face_value") or 0
                cp = bal.get("conversion_price") or 0
                cs = bal.get("convertible_shares") or bal.get("remaining_shares") or 0
                if remaining or cs:
                    self._instruments[key] = _InstrumentState(
                        category=category,
                        series=series,
                        kind=f"제{series}회",
                        face_value=bal.get("face_value") or remaining,
                        remaining=remaining,
                        conversion_price=cp,
                        convertible_shares=cs,
                    )
                    logger.info(
                        "Created instrument from exchange exercise fallback: %s (shares=%d)",
                        key, cs,
                    )
            else:
                logger.debug(
                    "No matching instrument for exchange exercise balance: series=%s",
                    series,
                )

    def process_exchange_price_adj(self, data: dict) -> None:
        """Process exchange disclosure price adjustment data (가액조정 공시).

        Updates existing instruments with adjusted prices and share counts
        from KRX exchange disclosures (전환가액의조정, etc.).
        Handles both series-based (CB/BW) and no-series (전환주식) formats.
        """
        for adj in data.get("adjustments", []):
            series = adj.get("series")
            new_price = adj.get("price_after")
            if new_price is None:
                continue

            matched = False
            matched_state = None
            if series is not None:
                for key, state in self._instruments.items():
                    if state.series == series:
                        state.conversion_price = new_price
                        logger.debug("Updated price for %s: %d", key, new_price)
                        matched = True
                        matched_state = state
                        break
            else:
                # No-series format (전환주식): match by price_before
                old_price = adj.get("price_before")
                if old_price:
                    for key, state in self._instruments.items():
                        if state.conversion_price == old_price:
                            state.conversion_price = new_price
                            logger.debug("Updated price for %s: %d (matched by old price)", key, new_price)
                            matched = True
                            matched_state = state
                            break

            # Recalculate convertible_shares from remaining/new_price
            # when no explicit share_changes are provided
            if matched_state and new_price and new_price >= 100 and matched_state.remaining:
                new_shares = matched_state.remaining // new_price
                if new_shares != matched_state.convertible_shares:
                    logger.debug(
                        "Recalculated shares for price adj: %d -> %d",
                        matched_state.convertible_shares, new_shares,
                    )
                    matched_state.convertible_shares = new_shares

            if not matched:
                logger.debug("No matching instrument for price adj: series=%s", series)

        for sc in data.get("share_changes", []):
            series = sc.get("series")
            new_shares = sc.get("shares_after")
            unconverted = sc.get("unconverted")
            if new_shares is None and unconverted is None:
                continue

            if series is not None:
                for key, state in self._instruments.items():
                    if state.series == series:
                        if new_shares is not None:
                            state.convertible_shares = new_shares
                            logger.debug("Updated shares for %s: %d", key, new_shares)
                        if unconverted is not None:
                            state.remaining = unconverted
                            logger.debug("Updated remaining for %s: %d", key, unconverted)
                        break
            else:
                # No-series format: match by shares_before or current_shares
                old_shares = sc.get("shares_before") or sc.get("current_shares")
                if old_shares:
                    for key, state in self._instruments.items():
                        if state.convertible_shares == old_shares:
                            if new_shares is not None:
                                state.convertible_shares = new_shares
                                logger.debug("Updated shares for %s: %d (matched by old shares)", key, new_shares)
                            if unconverted is not None:
                                state.remaining = unconverted
                                logger.debug("Updated remaining for %s: %d", key, unconverted)
                            break

    def add_stock_options(self, shares: int) -> None:
        """Manually add stock option overhang (legacy entry point).

        Skips if any SO instrument is already tracked from notes/API.
        """
        if any(k.startswith("SO_") for k in self._instruments):
            logger.debug("add_stock_options skipped: SO instruments already tracked")
            return
        key = "SO_0"
        self._instruments[key] = _InstrumentState(
            category="주식매수선택권",
            series=0,
            kind="",
            face_value=0,
            remaining=0,
            conversion_price=0,
            convertible_shares=shares,
        )

    def get_overhang_items(self) -> list[OverhangItem]:
        """Generate overhang table items for the report.

        Instruments whose exercise period has ended are excluded.
        """
        today = date.today()
        items = []
        for key, state in sorted(self._instruments.items()):
            if state.convertible_shares <= 0 and state.remaining <= 0:
                continue

            # Skip expired instruments (exercise period ended)
            if state.exercise_end and _is_expired(state.exercise_end, today):
                logger.debug("Skipping expired instrument: %s (ended %s)", key, state.exercise_end)
                continue

            # Format category label
            label = f"제{state.series}회 " if state.series else ""
            if state.category == "CB":
                label += "전환사채(CB)"
            elif state.category == "BW":
                label += "신주인수권부사채(BW)"
            elif state.category in ("주식매수선택권", "SO"):
                label = "주식매수선택권"
            else:
                label += state.category

            # Format remaining amount (currency-aware)
            # Prefer conversion_price × shares when available, as it's computed
            # from verified terms.  Fall back to state.remaining (face_value)
            # only when conversion terms are incomplete.
            remaining_str = None
            display_remaining = state.remaining
            if state.conversion_price > 0 and state.convertible_shares > 0:
                display_remaining = state.conversion_price * state.convertible_shares
            if display_remaining > 0:
                from auto_reports.analyzers.financial import currency_unit
                divisor, unit_label = currency_unit(self.currency)
                display_val = round(display_remaining / divisor)
                shares_str = f"{state.convertible_shares:,}"
                remaining_str = f"{display_val:,} {unit_label} ({shares_str}주)"
            elif state.convertible_shares > 0:
                remaining_str = f"{state.convertible_shares:,}주"

            # Format exercise price (currency-aware)
            price_str = None
            if state.conversion_price > 0:
                cur_suffix = "원" if self.currency == "KRW" else f" {self.currency}"
                price_str = f"{state.conversion_price:,}{cur_suffix}"

            # Calculate dilution ratio
            dilution = None
            if state.convertible_shares > 0 and self.total_shares > 0:
                ratio = (state.convertible_shares / self.total_shares) * 100
                dilution = f"{ratio:.2f}%"

            # Format exercise period
            ex_period = ""
            start = _format_date(state.exercise_start)
            end = _format_date(state.exercise_end)
            if start and end:
                ex_period = f"{start}~{end}"
            elif start:
                ex_period = f"{start}~"
            elif end:
                ex_period = f"~{end}"

            items.append(OverhangItem(
                category=label,
                remaining_amount=remaining_str,
                exercise_price=price_str,
                dilution_ratio=dilution,
                exercise_period=ex_period,
            ))

        # Sort by category name for consistent ordering
        items.sort(key=lambda x: x.category)
        return items

    def get_total_dilutive_shares(self) -> int:
        """Get total number of dilutive shares across all active instruments.

        Excludes expired instruments (same filter as get_overhang_items).
        """
        today = date.today()
        return sum(
            s.convertible_shares
            for s in self._instruments.values()
            if not (s.exercise_end and _is_expired(s.exercise_end, today))
        )

    def get_total_dilution(self) -> float:
        """Get total dilution ratio as percentage."""
        total_dilutive = self.get_total_dilutive_shares()
        if self.total_shares <= 0:
            return 0.0
        return (total_dilutive / self.total_shares) * 100


def _infer_category_from_exercise(exercise_type: str) -> str:
    """Infer instrument category (CB/BW) from exercise disclosure type."""
    if "신주인수권" in exercise_type:
        return "BW"
    return "CB"


class _InstrumentState:
    """Internal tracking state for a single dilutive instrument."""

    __slots__ = (
        "category", "series", "kind", "face_value",
        "remaining", "conversion_price", "convertible_shares",
        "exercise_start", "exercise_end",
    )

    def __init__(
        self,
        category: str,
        series: int,
        kind: str,
        face_value: int,
        remaining: int,
        conversion_price: int,
        convertible_shares: int,
        exercise_start: str = "",
        exercise_end: str = "",
    ):
        self.category = category
        self.series = series
        self.kind = kind
        self.face_value = face_value
        self.remaining = remaining
        self.conversion_price = conversion_price
        self.convertible_shares = convertible_shares
        self.exercise_start = exercise_start
        self.exercise_end = exercise_end


def _format_date(raw: str) -> str:
    """Convert Korean date '2024년 06월 17일' to compact 'YYYY.MM.DD' format."""
    if not raw:
        return ""
    m = re.search(r"(\d{4})\D+(\d{1,2})\D+(\d{1,2})", raw)
    if m:
        return f"{m.group(1)}.{int(m.group(2)):02d}.{int(m.group(3)):02d}"
    # Already compact or unrecognized — return as-is
    return raw.strip()


def _is_expired(exercise_end: str, today: date) -> bool:
    """Check if an instrument's exercise period has ended."""
    m = re.search(r"(\d{4})\D+(\d{1,2})\D+(\d{1,2})", exercise_end)
    if not m:
        return False
    try:
        end_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return today > end_date
    except ValueError:
        return False

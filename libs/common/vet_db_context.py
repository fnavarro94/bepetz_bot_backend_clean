# common/vet_db_context.py
# ==============================================================================
# Build LLM session context from a consultation_id (Anamnesis + Examen físico)
# Async, psycopg v3. Uses exam_catalog_id (your latest change).
# Env: PG_DSN="postgresql://user:pass@host:port/db"
# pip install "psycopg[binary]"
# ==============================================================================

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from psycopg.rows import dict_row
from psycopg import sql

PG_DSN_ENV = "PG_DSN"

# ── Introspection helpers ─────────────────────────────────────────────────────
async def _table_exists(conn: psycopg.AsyncConnection, name: str) -> bool:
    q = """
    SELECT EXISTS(
      SELECT 1 FROM information_schema.tables
      WHERE table_schema='public' AND table_name=%s
    ) AS exists
    """
    cur = await conn.execute(q, (name,))
    row = await cur.fetchone()
    return bool(row and row["exists"])

async def _column_exists(conn: psycopg.AsyncConnection, table: str, column: str) -> bool:
    q = """
    SELECT EXISTS(
      SELECT 1 FROM information_schema.columns
      WHERE table_schema='public' AND table_name=%s AND column_name=%s
    ) AS exists
    """
    cur = await conn.execute(q, (table, column))
    row = await cur.fetchone()
    return bool(row and row["exists"])

async def _specific_exam_detail_table(conn: psycopg.AsyncConnection) -> str:
    # Some DBs use "specific_re_exam_detail"; others "specific_exam_detail".
    if await _table_exists(conn, "specific_re_exam_detail"):
        return "specific_re_exam_detail"
    if await _table_exists(conn, "specific_exam_detail"):
        return "specific_exam_detail"
    return ""

async def _choose_det_columns(conn: psycopg.AsyncConnection) -> Tuple[Optional[str], Optional[str]]:
    # Handles typo "date_consulation" and alt reason column "reason_v_service_id"
    if await _column_exists(conn, "consultation_detail", "date_consultation"):
        date_col = "date_consultation"
    elif await _column_exists(conn, "consultation_detail", "date_consulation"):
        date_col = "date_consulation"
    else:
        date_col = None

    if await _column_exists(conn, "consultation_detail", "reason_vet_service_id"):
        reason_col = "reason_vet_service_id"
    elif await _column_exists(conn, "consultation_detail", "reason_v_service_id"):
        reason_col = "reason_v_service_id"
    else:
        reason_col = None

    return date_col, reason_col

# ── Small utils ───────────────────────────────────────────────────────────────
def _coerce_int(x: Any) -> Optional[int]:
    try:
        return None if x is None else int(x)
    except Exception:
        return None

def _map_or_str(val: Any, m: Optional[Dict[str, str]]) -> Optional[str]:
    if val is None:
        return None
    if m:
        s = str(val)
        return m.get(s, m.get(val, s))
    return str(val)

def _dedup_keep_order(items: List[str]) -> List[str]:
    seen, out = set(), []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out

def _derive_complaints(sigobs_items: List[Dict[str, Any]], signs_map: Optional[Dict[str, str]]) -> List[str]:
    out: List[str] = []
    for it in sigobs_items or []:
        if (it.get("section") or "").lower() != "signs":
            continue
        for cid in it.get("catalog_ids") or []:
            out.append(_map_or_str(cid, signs_map) or f"sign:{cid}")
        txt = (it.get("observations") or "").strip()
        if txt:
            out.append(txt)
    return _dedup_keep_order(out)

def _derive_exam_summary(spec_items: List[Dict[str, Any]], exam_system_map: Optional[Dict[str, str]]) -> str:
    # Uses exam_catalog_id as requested
    altered: List[str] = []
    for it in spec_items or []:
        if it is None:
            continue
        if it.get("is_normal") is False:
            altered.append(_map_or_str(it.get("exam_catalog_id"), exam_system_map) or str(it.get("exam_catalog_id")))
    return "Sistemas alterados: " + ", ".join(altered) if altered else "Sin hallazgos relevantes"

# ── Public: build session snapshot ────────────────────────────────────────────
async def build_session_snapshot_from_consultation(
    consultation_id: int,
    *,
    mucosa_map: Optional[Dict[str, str]] = None,
    duration_unit_map: Optional[Dict[str, str]] = None,
    reason_map: Optional[Dict[str, str]] = None,
    signs_map: Optional[Dict[str, str]] = None,
    exam_system_map: Optional[Dict[str, str]] = None,
    dsn: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Pulls vitals + signs/observations + specific exam + detail for a consultation,
    and returns the session_snapshot expected by the LLM generators.
    """
    dsn = dsn or os.getenv(PG_DSN_ENV)
    if not dsn:
        raise RuntimeError(f"{PG_DSN_ENV} is not set in the environment")

    async with await psycopg.AsyncConnection.connect(dsn, row_factory=dict_row) as conn:
        sed_table = await _specific_exam_detail_table(conn)
        date_col, reason_col = await _choose_det_columns(conn)

        # Vitals (latest row)
        q_vitals = """
            SELECT
                cvs.weight                    AS weight,
                cvs.weight_unit_catalog_id    AS weight_unit_catalog_id,
                cvs.temperature               AS temperature_c,
                cvs.heart_rate                AS heart_rate_bpm,
                cvs.respiration_rate          AS respiration_rate_bpm,
                cvs.body_condition            AS body_condition,
                cvs.tllc_seconds              AS tllc_seconds,
                cvs.mucosa_catalog_id         AS mucosa_catalog_id,
                cvs.saturation_percentage     AS spo2_percent,
                cvs.observation               AS vitals_observation
            FROM consultation_vital_signs cvs
            WHERE cvs.consultation_id = %s
            ORDER BY cvs.id DESC
            LIMIT 1
        """
        vcur = await conn.execute(q_vitals, (consultation_id,))
        vs = await vcur.fetchone() or {}

        # Detail (typo/variant-proof)
        q_det = sql.SQL("""
            SELECT
                {date_sel}        AS date_consultation,
                {reason_sel}      AS reason_vet_service_id,
                cd.duration_quantity,
                cd.duration_unit_catalog_id
            FROM consultation_detail cd
            WHERE cd.consultation_id = %s
            ORDER BY cd.id DESC
            LIMIT 1
        """).format(
            date_sel=sql.SQL("NULL") if not date_col else (sql.SQL("cd.") + sql.Identifier(date_col)),
            reason_sel=sql.SQL("NULL") if not reason_col else (sql.SQL("cd.") + sql.Identifier(reason_col)),
        )
        dcur = await conn.execute(q_det, (consultation_id,))
        det = await dcur.fetchone() or {}

        # Signs & observations
        q_sigobs = """
            SELECT
                jsonb_agg(
                    jsonb_build_object(
                        'catalog_ids', cso.catalog_ids,
                        'observations', cso.observations,
                        'section', cso.section
                    ) ORDER BY cso.id
                ) AS items
            FROM consultation_signs_observations cso
            WHERE cso.consultation_id = %s
        """
        scur = await conn.execute(q_sigobs, (consultation_id,))
        sigobs_items = (await scur.fetchone() or {}).get("items") or []

        # Specific exam (normal/alterado) — uses exam_catalog_id
        spec_items: List[Dict[str, Any]] = []
        if sed_table:
            q_spec = sql.SQL("""
                SELECT
                    jsonb_agg(
                        jsonb_build_object(
                            'exam_catalog_id', sed.exam_catalog_id,
                            'is_normal', sed.is_normal,
                            'specific_exam_id', sed.exam_catalog_id
                        ) ORDER BY sed.id
                    ) AS items
                FROM {sed} sed
                JOIN consultation_specific_exam cse
                  ON sed.exam_catalog_id = cse.id
                WHERE cse.consultation_id = %s
            """).format(sed=sql.Identifier(sed_table))
            spcur = await conn.execute(q_spec, (consultation_id,))
            spec_items = (await spcur.fetchone() or {}).get("items") or []

        # Consultation (for pet_id)
        q_consult = "SELECT id, pet_id FROM consultation WHERE id = %s"
        ccur = await conn.execute(q_consult, (consultation_id,))
        consult_row = await ccur.fetchone() or {}

    # Normalize to LLM fields
    vitals = {
        "temp_c": vs.get("temperature_c"),
        "hr_bpm": vs.get("heart_rate_bpm"),
        "rr_bpm": vs.get("respiration_rate_bpm"),
        "bcs": _coerce_int(vs.get("body_condition")),
        "mucous_membranes": _map_or_str(vs.get("mucosa_catalog_id"), mucosa_map),
        "tllc_seconds": vs.get("tllc_seconds"),
        "spo2_percent": vs.get("spo2_percent"),
        "weight_kg": vs.get("weight"),
        "vitals_notes": vs.get("vitals_observation"),
    }
    vitals = {k: v for k, v in vitals.items() if v is not None}

    complaints = _derive_complaints(sigobs_items, signs_map)
    exam_findings = _derive_exam_summary(spec_items, exam_system_map)

    history = {
        "reason": _map_or_str(det.get("reason_vet_service_id"), reason_map),
        "reason_code": det.get("reason_vet_service_id"),
        "duration": {
            "quantity": det.get("duration_quantity"),
            "unit": _map_or_str(det.get("duration_unit_catalog_id"), duration_unit_map),
            "unit_catalog_id": det.get("duration_unit_catalog_id"),
        },
        "date_consultation": det.get("date_consultation"),
    }

    return {
        "patient": {"id": consult_row.get("pet_id"), "weight_kg": vitals.get("weight_kg")},
        "complaints": complaints,
        "vitals": vitals,
        "exam": {"findings": exam_findings},
        "history": history,
        "labs": {},
    }



#--------------------------------------------
# Extractor de diagnostico definitivo
#--------------------------------------------
# common/vet_db_context.py  (append this near the bottom)

from typing import Optional, Dict
import psycopg
from psycopg.rows import dict_row

async def fetch_definitive_diagnosis(
    consultation_id: int | str,
    *,
    dsn: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Latest definitive diagnosis written by the vet for this consultation.
    Returns empty strings/None if not found.
    """
    dsn = dsn or os.getenv(PG_DSN_ENV)
    if not dsn:
        raise RuntimeError(f"{PG_DSN_ENV} is not set in the environment")

    async with await psycopg.AsyncConnection.connect(dsn, row_factory=dict_row) as conn:
        q = """
        SELECT
            cdd.final_diagnosis,
            cdd.condition_type_catalog_id,
            cdd.prognosis_type,
            cdd.observations
        FROM consultation_definitive_diagnosis cdd
        WHERE cdd.consultation_id = %s::bigint
        ORDER BY cdd.id DESC
        LIMIT 1
        """
        cur = await conn.execute(q, (consultation_id,))
        row = await cur.fetchone() or {}

    return {
        "final_diagnosis": row.get("final_diagnosis"),
        "condition_type_catalog_id": row.get("condition_type_catalog_id"),
        "prognosis_type": row.get("prognosis_type"),
        "observations": row.get("observations"),
    }




# ── Fetch prescribed medications for a consultation ───────────────────────────
from typing import Any, Dict, List, Optional
import psycopg
from psycopg.rows import dict_row

async def fetch_prescribed_medications(
    consultation_id: int | str,
    *,
    dsn: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Returns all medication rows stored for this consultation (what the vet prescribed).
    Raw catalog IDs are preserved; no mapping is performed.
    Shape per item (keys may be None depending on data):
      {
        "name", "active_ingredient", "dose_quantity", "dose_type_catalog_id",
        "presentation_catalog_id", "freq_med_catalog_id",
        "quantity", "quantity_med_catalog_id", "indications"
      }
    """
    dsn = dsn or os.getenv(PG_DSN_ENV)
    if not dsn:
        raise RuntimeError(f"{PG_DSN_ENV} is not set in the environment")

    async with await psycopg.AsyncConnection.connect(dsn, row_factory=dict_row) as conn:
        q = """
        SELECT
            cm.name,
            cm.active_ingredient,
            cm.dose_quantity,
            cm.dose_type_catalog_id,
            cm.presentation_catalog_id,
            cm.freq_med_catalog_id,
            cm.quantity,
            cm.quantity_med_catalog_id,
            cm.indications
        FROM consultation_medication cm
        WHERE cm.consultation_id = %s::bigint
        ORDER BY cm.id
        """
        cur = await conn.execute(q, (consultation_id,))
        rows = await cur.fetchall() or []

    return [dict(r) for r in rows]

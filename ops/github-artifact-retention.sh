#!/usr/bin/env bash
set -euo pipefail
REPOSITORY="${REPOSITORY:-${GITHUB_REPOSITORY:-}}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
MODE="${MODE:-dry-run}"
if [[ ! "$REPOSITORY" =~ ^[^/]+/[^/]+$ ]]; then echo "REPOSITORY must be owner/name" >&2; exit 2; fi
if [[ "$MODE" != "dry-run" && "$MODE" != "apply" ]]; then echo "MODE must be dry-run or apply" >&2; exit 2; fi
if ! [[ "$RETENTION_DAYS" =~ ^[0-9]+$ ]] || (( RETENTION_DAYS < 1 )); then echo "RETENTION_DAYS must be a positive integer" >&2; exit 2; fi
: "${GH_TOKEN:?GH_TOKEN is required}"
command -v gh >/dev/null; command -v jq >/dev/null; command -v date >/dev/null
cutoff_epoch="$(date -u -d "-${RETENTION_DAYS} days" +%s)"
cutoff_iso="$(date -u -d "@$cutoff_epoch" +%Y-%m-%dT%H:%M:%SZ)"
runs_json="$(gh api --paginate --slurp "repos/$REPOSITORY/actions/runs?status=success&per_page=100" | jq -c '[.[] | .workflow_runs[]? | select(.workflow_id != null and .created_at != null)]')"
keep_ids_json="$(jq -c 'sort_by(.workflow_id, .created_at) | group_by(.workflow_id) | map(max_by(.created_at).id)' <<<"$runs_json")"
declare -A keep_ids=()
while IFS= read -r run_id; do [[ -n "$run_id" ]] && keep_ids["$run_id"]=1; done < <(jq -r '.[]' <<<"$keep_ids_json")
artifacts_json="$(gh api --paginate --slurp "repos/$REPOSITORY/actions/artifacts?per_page=100" | jq -c '[.[] | .artifacts[]?]')"
scanned=0; eligible=0; deleted=0; preserved_latest=0; preserved_unknown=0; failed=0
echo "Repository: $REPOSITORY"; echo "Retention: $RETENTION_DAYS days"; echo "Cutoff (UTC): $cutoff_iso"; echo "Mode: $MODE"
while IFS= read -r artifact; do
 [[ -n "$artifact" ]] || continue
 scanned=$((scanned + 1))
 artifact_id="$(jq -r '.id // empty' <<<"$artifact")"
 artifact_name="$(jq -r '.name // "(без имени)"' <<<"$artifact")"
 created_at="$(jq -r '.created_at // empty' <<<"$artifact")"
 expired="$(jq -r '.expired // false' <<<"$artifact")"
 run_id="$(jq -r '.workflow_run.id // empty' <<<"$artifact")"
 if [[ -z "$artifact_id" || "$expired" == "true" || -z "$created_at" || -z "$run_id" ]]; then preserved_unknown=$((preserved_unknown + 1)); echo "PRESERVE artifact=${artifact_id:-unknown} reason=expired-or-missing-metadata"; continue; fi
 if ! created_epoch="$(date -u -d "$created_at" +%s 2>/dev/null)"; then preserved_unknown=$((preserved_unknown + 1)); echo "PRESERVE artifact=$artifact_id reason=invalid-created-at"; continue; fi
 if (( created_epoch >= cutoff_epoch )); then continue; fi
 if [[ -n "${keep_ids[$run_id]+present}" ]]; then preserved_latest=$((preserved_latest + 1)); echo "PRESERVE artifact=$artifact_id name=$artifact_name reason=latest-successful-run"; continue; fi
 run_state="$(gh api "repos/$REPOSITORY/actions/runs/$run_id" --jq '[.status, (.conclusion // "")] | @tsv')"
 run_status="${run_state%%$'\t'*}"; run_conclusion="${run_state#*$'\t'}"
 if [[ "$run_status" != "completed" || -z "$run_conclusion" ]]; then preserved_unknown=$((preserved_unknown + 1)); echo "PRESERVE artifact=$artifact_id reason=run-not-terminal"; continue; fi
 eligible=$((eligible + 1))
 if [[ "$MODE" == "dry-run" ]]; then echo "WOULD_DELETE artifact=$artifact_id name=$artifact_name created=$created_at run=$run_id conclusion=$run_conclusion"; continue; fi
 if gh api --method DELETE "repos/$REPOSITORY/actions/artifacts/$artifact_id" >/dev/null; then deleted=$((deleted + 1)); echo "DELETED artifact=$artifact_id name=$artifact_name created=$created_at"; else failed=$((failed + 1)); echo "FAILED artifact=$artifact_id name=$artifact_name" >&2; fi
done < <(jq -c '.[]' <<<"$artifacts_json")
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
 { echo "## Очистка GitHub Actions Artifacts"; echo; echo "- Репозиторий: `$REPOSITORY`"; echo "- Режим: `$MODE`"; echo "- Срок хранения: `$RETENTION_DAYS` дней"; echo "- Порог UTC: `$cutoff_iso`"; echo "- Проверено артефактов: **$scanned**"; echo "- Кандидатов старше срока: **$eligible**"; echo "- Сохранено для последних успешных запусков: **$preserved_latest**"; echo "- Сохранено из-за неполных/небезопасных метаданных: **$preserved_unknown**"; echo "- Удалено: **$deleted**"; echo "- Ошибок удаления: **$failed**"; } >> "$GITHUB_STEP_SUMMARY"
fi
echo "SUMMARY repository=$REPOSITORY scanned=$scanned eligible=$eligible deleted=$deleted preserved_latest=$preserved_latest preserved_unknown=$preserved_unknown failed=$failed"
(( failed == 0 ))

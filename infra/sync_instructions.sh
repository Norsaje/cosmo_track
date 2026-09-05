#!/usr/bin/env bash
# Детектор изменений в инструкциях команды.
#
# Зачем: teamlead и другие роли добавляют/меняют ТЗ и координационные документы без
# предупреждения. Промпт агента (CLAUDE.md, used_prompts/) обязан переписываться под них,
# иначе агент работает по устаревшим правилам. Скрипт сравнивает фактическое состояние
# инструкционных файлов во всех ветках с зафиксированным в infra/instructions.lock.
#
# Использование:
#   bash infra/sync_instructions.sh            # check: печатает дрейф, exit 1 если он есть
#   bash infra/sync_instructions.sh accept     # зафиксировать текущее состояние в lock
#   bash infra/sync_instructions.sh diff PATH  # что именно изменилось в файле с момента lock
#
# Коды выхода: 0 — инструкции актуальны; 1 — обнаружен дрейф; 2 — ошибка использования.

set -uo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "не git-репозиторий" >&2; exit 2; }
cd "$REPO_ROOT" || exit 2

LOCK="infra/instructions.lock"
# Ветки, где могут появляться инструкции. ED/ML/DL — рабочие ветки ролей: там ТЗ появляется
# раньше, чем попадает в main, и это ранний сигнал о будущем breaking change.
REFS=${INSTR_REFS:-"origin/main origin/backend origin/ML origin/DL origin/ED"}
# Что считается инструкцией: координационные файлы в корне, ролевые ТЗ и документы в docs/,
# промпты и сам CLAUDE.md.
PATTERN='^(CLAUDE\.md|AGENTS\.md|README\.md|[0-9]{2}_[^/]*\.md|[^/]*(coordination|instruction|prompt|task|spec)[^/]*\.md|docs/[^/]*\.(md|pdf)|docs/.*/[^/]*\.md|used_prompts/.*\.md|\.claude/.*\.md)$'

# Текущее фактическое состояние: ref, путь, sha блоба.
current_state() {
  local ref
  for ref in $REFS; do
    git rev-parse --verify --quiet "$ref" >/dev/null || continue
    git ls-tree -r "$ref" | awk '$2 == "blob" { sha=$3; $1=$2=$3=""; sub(/^\t?[ ]*/, ""); print sha "\t" $0 }' \
      | while IFS=$'\t' read -r sha path; do
          [[ "$path" =~ $PATTERN ]] && printf '%s\t%s\t%s\n' "$ref" "$path" "$sha"
        done
  done | sort
}

locked_state() {
  [ -f "$LOCK" ] && grep -v '^#' "$LOCK" | grep -v '^[[:space:]]*$' | sort
}

write_lock() {
  {
    echo "# infra/instructions.lock — зафиксированное состояние инструкций команды."
    echo "# Формат: <ref>\\t<path>\\t<blob-sha>. Обновляется ТОЛЬКО после актуализации"
    echo "# CLAUDE.md и used_prompts/backend_dev3_agent.md под новые инструкции."
    echo "# Зафиксировано: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    current_state
  } > "$LOCK"
}

cmd=${1:-check}

case "$cmd" in
  accept)
    write_lock
    echo "instructions.lock обновлён: $(grep -cv '^#' "$LOCK" | tr -d ' ') записей"
    ;;

  diff)
    path=${2:-}
    [ -n "$path" ] || { echo "использование: $0 diff <path> [ref]" >&2; exit 2; }
    only_ref=${3:-}
    shown=0
    # Один и тот же путь живёт в нескольких ветках, поэтому сравниваем по каждой ветке отдельно.
    while IFS=$'\t' read -r ref rpath rsha; do
      [ "$rpath" = "$path" ] || continue
      [ -z "$only_ref" ] || [ "$only_ref" = "$ref" ] || continue
      old=$(locked_state | awk -F'\t' -v r="$ref" -v p="$path" '$1 == r && $2 == p { print $3; exit }')
      if [ -z "$old" ]; then
        echo "=== $ref: НОВЫЙ файл $path — читать целиком (первые 200 строк) ==="
        git show "$rsha" | head -200; shown=1; continue
      fi
      [ "$old" = "$rsha" ] && continue
      echo "=== $ref: $path изменился ==="
      git diff --no-index --unified=3 <(git show "$old") <(git show "$rsha") \
        | sed -e "s|/dev/fd/[0-9]*|$path|g" -e "s|/proc/self/fd/[0-9]*|$path|g"
      shown=1
    done < <(current_state)
    [ "$shown" = 1 ] || echo "$path не менялся с момента lock ни в одной из веток: $REFS"
    ;;

  check)
    cur=$(current_state)
    lock=$(locked_state)

    if [ -z "$lock" ]; then
      echo "ДРЕЙФ: infra/instructions.lock отсутствует — инструкции ещё ни разу не зафиксированы."
      echo "$cur" | awk -F'\t' '{ print "  НОВЫЙ    " $2 "  (" $1 ")" }'
      exit 1
    fi

    new_files=$(comm -13 <(echo "$lock" | cut -f1,2 | sort) <(echo "$cur" | cut -f1,2 | sort))
    gone_files=$(comm -23 <(echo "$lock" | cut -f1,2 | sort) <(echo "$cur" | cut -f1,2 | sort))
    changed=$(join -t$'\t' -j1 -o 1.2,1.3,2.3 \
        <(echo "$lock" | awk -F'\t' '{ print $1"|"$2 "\t" $2 "\t" $3 }' | sort) \
        <(echo "$cur"  | awk -F'\t' '{ print $1"|"$2 "\t" $2 "\t" $3 }' | sort) \
      | awk -F'\t' '$2 != $3 { print $1 }' | sort -u)

    drift=0
    if [ -n "$new_files" ]; then
      drift=1; echo "НОВЫЕ файлы инструкций:"
      echo "$new_files" | awk -F'\t' '{ print "  + " $2 "  (" $1 ")" }'
    fi
    if [ -n "$changed" ]; then
      drift=1; echo "ИЗМЕНЁННЫЕ инструкции:"
      echo "$changed" | sed 's/^/  ~ /'
    fi
    if [ -n "$gone_files" ]; then
      drift=1; echo "УДАЛЁННЫЕ/ПЕРЕИМЕНОВАННЫЕ инструкции:"
      echo "$gone_files" | awk -F'\t' '{ print "  - " $2 "  (" $1 ")" }'
    fi

    if [ "$drift" = 1 ]; then
      echo
      echo "ТРЕБУЕТСЯ АКТУАЛИЗАЦИЯ ПРОМПТА (CLAUDE.md §1). Порядок:"
      echo "  1. bash infra/sync_instructions.sh diff <path>   — прочитать, что изменилось"
      echo "  2. переписать CLAUDE.md и used_prompts/backend_dev3_agent.md под новые правила"
      echo "  3. записать строку в журнал актуализации CLAUDE.md"
      echo "  4. bash infra/sync_instructions.sh accept"
      echo "  5. graphify . --update"
      exit 1
    fi
    echo "Инструкции актуальны: дрейфа нет."
    ;;

  *)
    echo "использование: $0 [check|accept|diff <path>]" >&2; exit 2 ;;
esac

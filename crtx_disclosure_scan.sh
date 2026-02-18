#!/usr/bin/env bash
# ============================================================================
# CRTX v1 Disclosure Scan & Cleanup Script
# ============================================================================
set -euo pipefail

CLEAN_MODE=false
if [[ "${1:-}" == "--clean" ]]; then
    CLEAN_MODE=true
    echo "⚠️  CLEAN MODE — flagged files will be removed"
    echo ""
fi

PASS=true
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

divider() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# ============================================================================
# 1. ALIA / DOMAIN KEYWORD SCAN
# ============================================================================
divider "1. ALIA & Domain Keyword Scan"

HARD_BLOCK_TERMS=(
    "insurance"
    "agency management"
    "\\bAMS\\b"
    "\\bbroker\\b"
    "underwriting"
    "\\bpremium\\b"
    "\\bCOI\\b"
    "\\brenewal\\b"
    "certificate of insurance"
    "binding trail"
    "client vault"
    "alia_patterns"
    "Atlas365"
    "Renewal Workbench"
    "ALIA Voice"
    "LiveKit"
    "Deepgram"
    "ElevenLabs"
    "Evidence v2"
    "canonical_json"
    "Connect threading"
    "THR-[0-9]"
    "Classification Intelligence"
    "VIN decode"
)

ALIA_PATTERN="\\bALIA\\b"

echo "Scanning tracked files for hard-block keywords..."
echo ""

KEYWORD_HITS=0

for term in "${HARD_BLOCK_TERMS[@]}"; do
    RESULTS=$(git ls-files -z | xargs -0 grep -rlin -E "$term" \
        --include='*.py' --include='*.md' --include='*.toml' \
        --include='*.yaml' --include='*.yml' --include='*.json' \
        --include='*.txt' --include='*.html' --include='*.js' \
        --include='*.jsx' --include='*.ts' --include='*.tsx' \
        --include='*.cfg' --include='*.ini' --include='*.svg' \
        2>/dev/null || true)

    if [[ -n "$RESULTS" ]]; then
        echo -e "${RED}🔴 FOUND: \"$term\"${NC}"
        echo "$RESULTS" | while read -r file; do
            LINE=$(grep -n -i -E "$term" "$file" 2>/dev/null | head -3)
            echo "   $file"
            echo "$LINE" | sed 's/^/      /'
        done
        echo ""
        KEYWORD_HITS=$((KEYWORD_HITS + 1))
        PASS=false
    fi
done

ALIA_RESULTS=$(git ls-files -z | xargs -0 grep -rln -E "$ALIA_PATTERN" \
    --include='*.py' --include='*.md' --include='*.toml' \
    --include='*.yaml' --include='*.yml' --include='*.json' \
    --include='*.txt' --include='*.html' --include='*.js' \
    --include='*.jsx' --include='*.ts' --include='*.tsx' \
    --include='*.cfg' --include='*.ini' --include='*.svg' \
    2>/dev/null || true)

if [[ -n "$ALIA_RESULTS" ]]; then
    echo -e "${RED}🔴 FOUND: \"ALIA\" (standalone word)${NC}"
    echo "$ALIA_RESULTS" | while read -r file; do
        LINE=$(grep -n -E "$ALIA_PATTERN" "$file" 2>/dev/null | head -3)
        echo "   $file"
        echo "$LINE" | sed 's/^/      /'
    done
    echo ""
    KEYWORD_HITS=$((KEYWORD_HITS + 1))
    PASS=false
fi

if [[ $KEYWORD_HITS -eq 0 ]]; then
    echo -e "${GREEN}✅ No hard-block keywords found${NC}"
fi

# ============================================================================
# 2. FUTURE TIER / PATENT-SENSITIVE TERMS
# ============================================================================
divider "2. Future Tier & Patent-Sensitive Terms"

TIER_TERMS=(
    "Brain"
    "Collective Intelligence"
    "\\$49"
    "\\$149"
    "\\$29"
    "observation schema"
    "project_knowledge\\.json"
)

TIER_HITS=0

for term in "${TIER_TERMS[@]}"; do
    RESULTS=$(git ls-files -z | xargs -0 grep -rln -E "$term" \
        --include='*.py' --include='*.md' --include='*.toml' \
        --include='*.yaml' --include='*.yml' --include='*.json' \
        --include='*.txt' --include='*.html' --include='*.js' \
        --include='*.jsx' --include='*.ts' --include='*.tsx' \
        2>/dev/null || true)

    if [[ -n "$RESULTS" ]]; then
        echo -e "${YELLOW}🟡 REVIEW: \"$term\"${NC}"
        echo "$RESULTS" | while read -r file; do
            LINE=$(grep -n -E "$term" "$file" 2>/dev/null | head -3)
            echo "   $file"
            echo "$LINE" | sed 's/^/      /'
        done
        echo ""
        TIER_HITS=$((TIER_HITS + 1))
    fi
done

if [[ $TIER_HITS -eq 0 ]]; then
    echo -e "${GREEN}✅ No future-tier terms found${NC}"
else
    echo -e "${YELLOW}⚠️  $TIER_HITS term(s) flagged for manual review${NC}"
    echo "   These may be false positives. Check context before removing."
fi

# ============================================================================
# 3. FILES THAT SHOULDN'T BE IN THE PUBLIC REPO
# ============================================================================
divider "3. Files / Folders to Remove"

REMOVE_TARGETS=(
    "api"
    "supabase"
    "package-lock.json"
    "triad-site"
)

REMOVE_HITS=0

for target in "${REMOVE_TARGETS[@]}"; do
    if [[ -e "$target" ]]; then
        if [[ -d "$target" ]]; then
            FILE_COUNT=$(find "$target" -type f | wc -l)
            echo -e "${RED}🔴 FOUND: $target/ ($FILE_COUNT files)${NC}"
        else
            echo -e "${RED}🔴 FOUND: $target${NC}"
        fi

        if [[ "$CLEAN_MODE" == true ]]; then
            if git ls-files --error-unmatch "$target" &>/dev/null 2>&1 || \
               git ls-files --error-unmatch "$target/" &>/dev/null 2>&1; then
                echo -e "   ${YELLOW}Removing from git tracking...${NC}"
                git rm -r --cached "$target" 2>/dev/null || true
            fi
            echo -e "   ${YELLOW}Deleting from disk...${NC}"
            rm -rf "$target"
            echo -e "   ${GREEN}Removed.${NC}"
        else
            echo "   → Run with --clean to remove, or manually: git rm -r $target"
        fi
        echo ""
        REMOVE_HITS=$((REMOVE_HITS + 1))
        PASS=false
    fi
done

if [[ $REMOVE_HITS -eq 0 ]]; then
    echo -e "${GREEN}✅ No flagged files/folders found${NC}"
fi

# ============================================================================
# 4. .GITIGNORE CHECK
# ============================================================================
divider "4. .gitignore Verification"

GITIGNORE_REQUIRED=(
    "crtx-output/"
    "triad-output/"
    ".env"
    "*.env"
    "config/domain/"
)

if [[ -f ".gitignore" ]]; then
    MISSING_IGNORES=0
    for pattern in "${GITIGNORE_REQUIRED[@]}"; do
        if grep -qF "$pattern" .gitignore 2>/dev/null; then
            echo -e "${GREEN}  ✅ $pattern${NC}"
        else
            echo -e "${RED}  🔴 MISSING: $pattern${NC}"
            MISSING_IGNORES=$((MISSING_IGNORES + 1))
            PASS=false
        fi
    done

    if [[ $MISSING_IGNORES -gt 0 ]]; then
        echo ""
        echo "   Add missing patterns to .gitignore before making repo public."
    fi
else
    echo -e "${RED}🔴 No .gitignore found!${NC}"
    PASS=false
fi

# ============================================================================
# 5. BRANDING CHECK
# ============================================================================
divider "5. Branding Check — Triad Orchestrator References"

BRAND_FILES=(
    "README.md"
    "CONTRIBUTING.md"
    "CLA.md"
    "CLAUDE.md"
    "pyproject.toml"
    "CHANGELOG.md"
)

BRAND_HITS=0

for file in "${BRAND_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        RESULTS=$(grep -n -i "triad orchestrator" "$file" 2>/dev/null || true)
        if [[ -n "$RESULTS" ]]; then
            echo -e "${YELLOW}🟡 \"Triad Orchestrator\" in $file:${NC}"
            echo "$RESULTS" | head -5 | sed 's/^/      /'
            echo ""
            BRAND_HITS=$((BRAND_HITS + 1))
        fi
    fi
done

PY_BRAND=$(git ls-files -z '*.py' | xargs -0 grep -rln -i "triad orchestrator" 2>/dev/null || true)
if [[ -n "$PY_BRAND" ]]; then
    echo -e "${YELLOW}🟡 \"Triad Orchestrator\" in Python source:${NC}"
    echo "$PY_BRAND" | while read -r file; do
        LINE=$(grep -n -i "triad orchestrator" "$file" 2>/dev/null | head -3)
        echo "   $file"
        echo "$LINE" | sed 's/^/      /'
    done
    echo ""
    BRAND_HITS=$((BRAND_HITS + 1))
fi

if [[ $BRAND_HITS -eq 0 ]]; then
    echo -e "${GREEN}✅ No 'Triad Orchestrator' references in user-facing files${NC}"
else
    echo -e "${YELLOW}⚠️  $BRAND_HITS location(s) still reference 'Triad Orchestrator'${NC}"
    echo "   Replace with 'CRTX' in user-facing contexts. 'triad/' module name is OK."
fi

# ============================================================================
# 6. SECRETS CHECK
# ============================================================================
divider "6. Secrets Check (current files only)"

SECRET_PATTERNS=(
    "sk-[a-zA-Z0-9]\\{20,\\}"
    "ANTHROPIC_API_KEY.*=.*sk-"
    "OPENAI_API_KEY.*=.*sk-"
    "xai-[a-zA-Z0-9]\\{20,\\}"
    "AIza[a-zA-Z0-9]\\{30,\\}"
)

SECRET_HITS=0

for pattern in "${SECRET_PATTERNS[@]}"; do
    RESULTS=$(git ls-files -z | xargs -0 grep -rln "$pattern" 2>/dev/null || true)
    if [[ -n "$RESULTS" ]]; then
        echo -e "${RED}🔴 POSSIBLE SECRET: pattern \"$pattern\"${NC}"
        echo "$RESULTS" | sed 's/^/   /'
        echo ""
        SECRET_HITS=$((SECRET_HITS + 1))
        PASS=false
    fi
done

if [[ $SECRET_HITS -eq 0 ]]; then
    echo -e "${GREEN}✅ No secrets detected in tracked files${NC}"
fi

echo ""
echo -e "${YELLOW}Note: This only scans current files. For full git history secret scan:${NC}"
echo "  git log --all -p | grep -E 'sk-[a-zA-Z0-9]{20,}|xai-[a-zA-Z0-9]{20,}'"

# ============================================================================
# 7. PROMPT TEMPLATE SCAN
# ============================================================================
divider "7. Prompt Template Scan"

PROMPT_DIR="triad/prompts"
if [[ -d "$PROMPT_DIR" ]]; then
    PROMPT_HITS=0
    for prompt_file in "$PROMPT_DIR"/*.md; do
        if [[ -f "$prompt_file" ]]; then
            ISSUES=$(grep -n -i -E "insurance|ALIA|agency|broker|AMS|atlas365|renewal|premium|COI|binding trail|evidence v2" "$prompt_file" 2>/dev/null || true)
            if [[ -n "$ISSUES" ]]; then
                echo -e "${RED}🔴 $(basename "$prompt_file"):${NC}"
                echo "$ISSUES" | sed 's/^/      /'
                echo ""
                PROMPT_HITS=$((PROMPT_HITS + 1))
                PASS=false
            else
                echo -e "${GREEN}  ✅ $(basename "$prompt_file")${NC}"
            fi
        fi
    done

    if [[ $PROMPT_HITS -eq 0 ]]; then
        echo -e "${GREEN}✅ All prompt templates clean${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Prompt directory $PROMPT_DIR not found${NC}"
fi

# ============================================================================
# 8. ENV / CONFIG FILE CHECK
# ============================================================================
divider "8. Environment & Config Files"

ENV_FILES=$(git ls-files | grep -E '\.env$' 2>/dev/null || true)
if [[ -n "$ENV_FILES" ]]; then
    echo -e "${RED}🔴 .env files tracked in git:${NC}"
    echo "$ENV_FILES" | sed 's/^/   /'
    PASS=false
else
    echo -e "${GREEN}✅ No .env files tracked${NC}"
fi

# ============================================================================
# SUMMARY
# ============================================================================
divider "SCAN COMPLETE"

if [[ "$PASS" == true ]]; then
    echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✅  ALL CHECKS PASSED — READY TO SHIP  ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
else
    echo -e "${RED}╔══════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ❌  ISSUES FOUND — FIX BEFORE SHIPPING  ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════╝${NC}"
    echo ""
    echo "Fix the 🔴 items above, then re-run this script."
    echo "Review the 🟡 items manually for false positives."
    echo ""
    echo "To auto-remove flagged files/folders:"
    echo "  ./crtx_disclosure_scan.sh --clean"
fi

echo ""

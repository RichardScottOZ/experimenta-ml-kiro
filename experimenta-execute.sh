#!/bin/bash

# Experimenta ML Execution Loop for Kiro CLI
# Adapted from ralph-kiro's ralph-execute.sh for ML experimentation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
MAX_ITERATIONS=50
COMPLETION_WORD="DONE"
PROMPT_FILE="PROMPT.md"
LOG_FILE="experimenta-execution.log"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --completion-word)
            COMPLETION_WORD="$2"
            shift 2
            ;;
        --prompt-file)
            PROMPT_FILE="$2"
            shift 2
            ;;
        --log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Experimenta ML Execution Loop - Run the autonomous ML experimentation phase"
            echo ""
            echo "Usage: ./experimenta-execute.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --max-iterations N    Maximum iterations (default: 50)"
            echo "  --completion-word W   Word that signals completion (default: DONE)"
            echo "  --prompt-file F       Prompt file to use (default: PROMPT.md)"
            echo "  --log-file L          Log file path (default: experimenta-execution.log)"
            echo "  --help                Show this help message"
            echo ""
            echo "Example:"
            echo "  ./experimenta-execute.sh --max-iterations 30"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if PROMPT.md exists
if [ ! -f "$PROMPT_FILE" ]; then
    echo -e "${RED}Error: $PROMPT_FILE not found${NC}"
    echo "Make sure you've run the plan phase first:"
    echo "  kiro-cli chat --agent ml-plan"
    exit 1
fi

# Check if kiro-cli is installed
if ! command -v kiro-cli &> /dev/null; then
    echo -e "${RED}Error: kiro-cli not found${NC}"
    echo "Please install Kiro CLI: https://kiro.dev/cli/"
    exit 1
fi

# Check if mission.yaml exists
if [ ! -f "mission.yaml" ]; then
    echo -e "${YELLOW}Warning: mission.yaml not found${NC}"
    echo "The ML experiment agent needs mission.yaml to run."
fi

# Check if PROGRAM.md exists
if [ ! -f "PROGRAM.md" ]; then
    echo -e "${YELLOW}Warning: PROGRAM.md not found${NC}"
    echo "The ML experiment agent needs PROGRAM.md as its behavioral contract."
fi

# Initialize log file
echo "Experimenta ML Execution Log - $(date)" > "$LOG_FILE"
echo "Max Iterations: $MAX_ITERATIONS" >> "$LOG_FILE"
echo "Completion Word: $COMPLETION_WORD" >> "$LOG_FILE"
echo "Prompt File: $PROMPT_FILE" >> "$LOG_FILE"
echo "----------------------------------------" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Display banner
echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Experimenta ML - Kiro CLI Execution Loop    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Max Iterations: ${GREEN}$MAX_ITERATIONS${NC}"
echo -e "  Completion Word: ${GREEN}$COMPLETION_WORD${NC}"
echo -e "  Prompt File: ${GREEN}$PROMPT_FILE${NC}"
echo -e "  Log File: ${GREEN}$LOG_FILE${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the loop at any time${NC}"
echo ""

# Confirmation
read -p "Ready to start ML experimentation? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Execution cancelled${NC}"
    exit 0
fi

# Main execution loop
iteration=0
start_time=$(date +%s)

echo -e "${GREEN}Starting ML experimentation loop...${NC}"
echo ""

while [ $iteration -lt $MAX_ITERATIONS ]; do
    iteration=$((iteration + 1))
    iteration_start=$(date +%s)

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Iteration $iteration/$MAX_ITERATIONS${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Log iteration start
    echo "=== Iteration $iteration - $(date) ===" >> "$LOG_FILE"

    # Execute kiro-cli with the prompt
    output=$(cat "$PROMPT_FILE" | kiro-cli chat --no-interactive -a 2>&1) || {
        echo -e "${RED}Error: kiro-cli command failed${NC}"
        echo "$output"
        echo "ERROR: kiro-cli failed at iteration $iteration" >> "$LOG_FILE"
        echo "$output" >> "$LOG_FILE"
        exit 1
    }

    # Display output
    echo "$output"

    # Log output
    echo "$output" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"

    # Check for completion
    if echo "$output" | grep -q "$COMPLETION_WORD"; then
        iteration_end=$(date +%s)
        total_time=$((iteration_end - start_time))

        echo ""
        echo -e "${GREEN}╔════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║     ML EXPERIMENTATION COMPLETE! 🎉            ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${YELLOW}Summary:${NC}"
        echo -e "  Completed at iteration: ${GREEN}$iteration${NC}"
        echo -e "  Total time: ${GREEN}$((total_time / 60))m $((total_time % 60))s${NC}"
        echo -e "  Log file: ${GREEN}$LOG_FILE${NC}"
        echo ""
        echo -e "${YELLOW}Check outputs/ for results:${NC}"
        echo -e "  final_experiment.py  — winning experiment code"
        echo -e "  final_pipeline.pkl   — fitted pipeline"
        echo -e "  report.md            — full run summary"
        echo ""

        echo "=== COMPLETED at iteration $iteration ===" >> "$LOG_FILE"
        echo "Total time: $total_time seconds" >> "$LOG_FILE"

        exit 0
    fi

    # Check if stuck
    if echo "$output" | grep -q "STUCK"; then
        echo ""
        echo -e "${RED}╔════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║              EXECUTION STUCK                   ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${YELLOW}The agent reported being stuck. Review the output above.${NC}"
        echo -e "${YELLOW}Consider:${NC}"
        echo -e "  1. Adding more specific guardrails to PROMPT.md"
        echo -e "  2. Checking mission.yaml domain_knowledge section"
        echo -e "  3. Reviewing error_analysis/ for patterns"
        echo -e "  4. Fixing the issue manually and resuming"
        echo ""

        echo "=== STUCK at iteration $iteration ===" >> "$LOG_FILE"
        exit 1
    fi

    # Calculate iteration time
    iteration_end=$(date +%s)
    iteration_time=$((iteration_end - iteration_start))
    echo -e "${YELLOW}Iteration time: ${iteration_time}s${NC}"
    echo ""

    # Small delay to avoid rate limiting
    sleep 2
done

# Max iterations reached
total_time=$(($(date +%s) - start_time))

echo ""
echo -e "${YELLOW}╔════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║     MAX ITERATIONS REACHED                     ║${NC}"
echo -e "${YELLOW}╚════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Summary:${NC}"
echo -e "  Iterations completed: ${GREEN}$MAX_ITERATIONS${NC}"
echo -e "  Total time: ${GREEN}$((total_time / 60))m $((total_time % 60))s${NC}"
echo -e "  Log file: ${GREEN}$LOG_FILE${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Review TODO.md to see progress"
echo -e "  2. Check experiments.json for results"
echo -e "  3. Adjust PROMPT.md guardrails if needed"
echo -e "  4. Run again: ./experimenta-execute.sh"
echo ""

echo "=== MAX ITERATIONS REACHED ===" >> "$LOG_FILE"
echo "Total time: $total_time seconds" >> "$LOG_FILE"

exit 1

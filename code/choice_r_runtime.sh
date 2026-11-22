# Sourced by run_single_trial_choice.sh. On success, rr is the verified R command.
choice_r_works() {
    "$@" -e 'suppressPackageStartupMessages(library(lme4)); m <- glmer(cbind(incidence, size-incidence) ~ period + (1|herd), data=cbpp, family=binomial); stopifnot(is.finite(as.numeric(logLik(m))))' >/dev/null 2>&1
}

choice_conda_r() {
    local prefix=$1
    rr=(conda run --no-capture-output -p "$prefix" env -u R_HOME -u R_LIBS -u R_LIBS_SITE R_LIBS_USER=NULL "$prefix/bin/Rscript" --vanilla)
    [[ -x "$prefix/bin/Rscript" ]] && choice_r_works "${rr[@]}"
}

select_choice_r() {
    local prefix runtime
    if [[ -n ${SRNDNA_CHOICE_RSCRIPT:-} ]]; then
        rr=("$SRNDNA_CHOICE_RSCRIPT")
        if choice_r_works "${rr[@]}"; then return 0; fi
        echo "The configured SRNDNA_CHOICE_RSCRIPT could not fit the lme4 test model." >&2
        return 1
    fi
    rr=(Rscript)
    if choice_r_works "${rr[@]}"; then return 0; fi
    if ! command -v conda >/dev/null; then
        echo "A working Rscript with lme4 or conda is required." >&2
        return 1
    fi
    runtime="${XDG_CACHE_HOME:-$HOME/.cache}/srndna-ultimatum/choice-r-4.5.2-lme4-2.0.1"
    if choice_conda_r "$runtime"; then return 0; fi
    while IFS= read -r prefix; do
        if [[ "$prefix" != "$runtime" ]] && choice_conda_r "$prefix"; then return 0; fi
    done < <(conda env list --json | python3 -c 'import json,sys; print("\n".join(json.load(sys.stdin)["envs"]))')
    echo "Preparing a user-owned R 4.5.2 environment with lme4 2.0.1 in $runtime" >&2
    mkdir -p "$(dirname "$runtime")"
    (
        if command -v flock >/dev/null; then flock -x 9 || exit 1; fi
        if choice_conda_r "$runtime"; then exit 0; fi
        local action=create
        if [[ -f "$runtime/conda-meta/history" ]]; then action=install; fi
        conda "$action" --yes --prefix "$runtime" --override-channels -c conda-forge 'r-base=4.5.2' 'r-lme4=2.0_1' || exit 1
    ) 9>"$runtime.lock" || return 1
    if choice_conda_r "$runtime"; then return 0; fi
    echo "R was installed but the mixed-model test failed. R's diagnostic follows." >&2
    "${rr[@]}" -e 'library(lme4); print(glmer(cbind(incidence,size-incidence) ~ period + (1|herd), data=cbpp, family=binomial))' >&2
    return 1
}

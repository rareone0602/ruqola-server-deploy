#!/bin/bash
# The project's own rules, so a new script cannot be added half-way.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

t "Every bin/ script is executable bash"
for f in "$ROOT"/bin/*; do
    n=$(basename "$f")
    check "$n is executable" "$([[ -x "$f" ]] && echo yes || echo no)" "yes"
    check "$n starts with #!/bin/bash" "$(head -1 "$f")" "#!/bin/bash"
    bash -n "$f"; check "$n parses" "$?" "0"
done

t "Every bin/ script is in the MANIFEST, and every MANIFEST source exists"
for f in "$ROOT"/bin/*; do
    n=$(basename "$f")
    check "MANIFEST installs bin/$n" "$(grep -cE "^file[[:space:]]+bin/$n[[:space:]]" "$ROOT/MANIFEST")" "1"
done
while read -r kind src _; do
    [[ "$kind" =~ ^(file|unit|render)$ ]] || continue
    check "MANIFEST source exists: $src" "$(present "$ROOT/$src")" "present"
done < <(grep -vE '^\s*(#|$)' "$ROOT/MANIFEST")

t "Every bin/ script has a test that names it"
for f in "$ROOT"/bin/*; do
    n=$(basename "$f")
    check "some tests/test_*.sh mentions $n" "$(( $(grep -l "$n" "$ROOT"/tests/test_*.sh | grep -vc test_project_layout) > 0 ))" "1"
done

t "Scripts that use the shared library all carry the same loader block"
canon=$(sed -n '/^_here=/,/^source "\$RUQOLA_ADMIN_LIB\/init.sh"/p' "$ROOT/bin/scratch-cleanup.sh" | sed 's/[a-z_-]*\.sh: cannot load/X: cannot load/')
check "canonical loader found in scratch-cleanup.sh" "$(( $(wc -l <<<"$canon") >= 5 ))" "1"
for f in "$ROOT"/bin/*; do
    grep -q 'RUQOLA_ADMIN_LIB' "$f" || continue
    this=$(sed -n '/^_here=/,/^source "\$RUQOLA_ADMIN_LIB\/init.sh"/p' "$f" | sed 's/[a-z_-]*\.sh: cannot load/X: cannot load/')
    check "$(basename "$f") uses the canonical loader" "$([[ "$this" == "$canon" ]] && echo same || echo differs)" "same"
done

t "Root safety: the suite refuses root, and the installer drops privileges for it"
check "tests/lib.sh exits 2 when EUID is 0" "$(grep -c 'EUID == 0' "$ROOT/tests/lib.sh")" "1"
check "install.sh runs the tests as SUDO_USER when root" "$(grep -c 'runuser -u "\$SUDO_USER"' "$ROOT/install.sh")" "1"
check "install.sh refuses a bare root login for the tests" "$(grep -c 'no SUDO_USER to drop to' "$ROOT/install.sh")" "1"

t "lib/init.sh loads every lib/*.sh file"
for f in "$ROOT"/lib/*.sh; do
    n=$(basename "$f" .sh); [[ "$n" == init ]] && continue
    check "init.sh lists $n" "$(grep -cE "for _ruqola_f in .*\b$n\b" "$ROOT/lib/init.sh")" "1"
done

finish

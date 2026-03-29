# Security

## Known transitive dependency: diskcache (CVE-2025-69872)

**diskcache** is a transitive dependency of:

- **llama-cpp-python** (optional backend: `--extra llamacpp`)
- **outlines** → pulled in by **mlx-openai-server** (optional backend: `--extra mlx`)

[Advisory](https://nvd.nist.gov/vuln/detail/CVE-2025-69872): diskcache ≤5.6.3 uses Python object serialization by default in a way that can deserialize attacker-controlled data from the cache directory. An attacker with **write access to the cache directory** can achieve arbitrary code execution when the application reads from the cache.

**Status:** No patched version has been released by the diskcache maintainer. The last release is 5.6.3 (Aug 2023).

### Mitigation

- **Restrict cache directory permissions:** Ensure any cache directories used by the llama-cpp-python or mlx-openai-server backends are only writable by the process running the server (e.g. not world-writable, not in a shared/temp location that untrusted users can write to).
- **Deployment:** When running in Docker or on a host, run the server as a dedicated user and point caches to a directory owned by that user with mode `700` or equivalent.
- If you do not use the `llamacpp` or `mlx` optional backends, install without them so diskcache is not pulled in:
  ```bash
  uv sync   # no --extra mlx --extra llamacpp
  ```

### Dependabot (diskcache)

When dismissing the Dependabot alert for diskcache in the GitHub Security tab, you can use:

- **Reason:** "No patch available"
- **Comment:** "Transitive dependency of llama-cpp-python and outlines (via mlx-openai-server). Risk limited to attackers with write access to cache directory. Mitigation and status documented in SECURITY.md."

## Known transitive dependency: pygments (CVE-2026-4539)

**pygments** is a transitive dependency of:

- **rich** (used for CLI output, e.g. `benchmark_models.py`)
- **pytest** (development dependency)

[Advisory](https://github.com/advisories/GHSA-5239-wwwm-4pmq): pygments ≤2.19.2 has inefficient regex complexity in `AdlLexer` (ReDoS). The advisory describes **local** attack prerequisites and **low** impact (availability).

**Status:** No patched release on PyPI yet; latest is still 2.19.2. Track [pygments/pygments#3058](https://github.com/pygments/pygments/issues/3058).

### Mitigation

- Normal server and test usage does not feed untrusted input through the affected lexer path; risk is low for typical deployments.
- After a fixed release appears on PyPI, run `uv lock` (and bump constraints if needed) so Dependabot clears the alert.

### Dependabot (pygments)

When dismissing the Dependabot alert for pygments until a release fix exists:

- **Reason:** "No patch available"
- **Comment:** "Transitive via rich and pytest. CVE-2026-4539 ReDoS in AdlLexer; no PyPI release beyond 2.19.2 yet. See SECURITY.md and https://github.com/pygments/pygments/issues/3058."

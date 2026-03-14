# Security

## Known transitive dependency: diskcache (CVE-2025-69872)

**diskcache** is a transitive dependency of:

- **llama-cpp-python** (optional backend: `--extra llamacpp`)
- **outlines** → pulled in by **mlx-openai-server** (optional backend: `--extra mlx`)

[Advisory](https://nvd.nist.gov/vuln/detail/CVE-2025-69872): diskcache ≤5.6.3 uses Python pickle for serialization by default. An attacker with **write access to the cache directory** can achieve arbitrary code execution when the application reads from the cache.

**Status:** No patched version has been released by the diskcache maintainer. The last release is 5.6.3 (Aug 2023).

### Mitigation

- **Restrict cache directory permissions:** Ensure any cache directories used by the llama-cpp-python or mlx-openai-server backends are only writable by the process running the server (e.g. not world-writable, not in a shared/temp location that untrusted users can write to).
- **Deployment:** When running in Docker or on a host, run the server as a dedicated user and point caches to a directory owned by that user with mode `700` or equivalent.
- If you do not use the `llamacpp` or `mlx` optional backends, install without them so diskcache is not pulled in:
  ```bash
  uv sync   # no --extra mlx --extra llamacpp
  ```

### Dependabot alert #5

When dismissing the Dependabot alert for diskcache in the GitHub Security tab, you can use:

- **Reason:** "No patch available"
- **Comment:** "Transitive dependency of llama-cpp-python and outlines (via mlx-openai-server). Risk limited to attackers with write access to cache directory. Mitigation and status documented in SECURITY.md."

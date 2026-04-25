"""Tests for PGOA Slurm binding actions."""

from __future__ import annotations

import unittest

from claw_backend.pgoa.actions import (
    VALID_CPU_BIND,
    VALID_MEM_BIND,
    apply_slurm_binding,
    validate_slurm_binding_params,
)

_SAMPLE_SCRIPT = """\
#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8

module load openmpi
srun ./my_app
"""


class TestValidateSlurmBindingParams(unittest.TestCase):
    def test_valid_params(self):
        errors = validate_slurm_binding_params(
            ntasks_per_node=8,
            cpus_per_task=8,
            mem_bind="local",
            cpu_bind="cores",
        )
        self.assertEqual(errors, [])

    def test_invalid_ntasks_zero(self):
        errors = validate_slurm_binding_params(ntasks_per_node=0)
        self.assertTrue(len(errors) > 0)

    def test_invalid_ntasks_negative(self):
        errors = validate_slurm_binding_params(ntasks_per_node=-1)
        self.assertTrue(len(errors) > 0)

    def test_invalid_mem_bind(self):
        errors = validate_slurm_binding_params(mem_bind="invalid_value")
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("mem_bind" in e for e in errors))

    def test_invalid_cpu_bind(self):
        errors = validate_slurm_binding_params(cpu_bind="invalid_value")
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("cpu_bind" in e for e in errors))

    def test_all_none_valid(self):
        errors = validate_slurm_binding_params()
        self.assertEqual(errors, [])

    def test_valid_mem_bind_values(self):
        for v in VALID_MEM_BIND:
            errors = validate_slurm_binding_params(mem_bind=v)
            self.assertEqual(errors, [], f"Expected {v!r} to be valid")

    def test_valid_cpu_bind_values(self):
        for v in VALID_CPU_BIND:
            errors = validate_slurm_binding_params(cpu_bind=v)
            self.assertEqual(errors, [], f"Expected {v!r} to be valid")


class TestApplySlurmBinding(unittest.TestCase):
    def test_replace_existing_ntasks_per_node(self):
        new_script, changes = apply_slurm_binding(
            _SAMPLE_SCRIPT, ntasks_per_node=16
        )
        self.assertIn("--ntasks-per-node=16", new_script)
        self.assertNotIn("--ntasks-per-node=8", new_script)
        self.assertTrue(any("replaced" in c for c in changes))

    def test_add_new_mem_bind(self):
        new_script, changes = apply_slurm_binding(
            _SAMPLE_SCRIPT, mem_bind="local"
        )
        self.assertIn("--mem-bind=local", new_script)
        self.assertTrue(any("added" in c for c in changes))

    def test_replace_existing_cpus_per_task(self):
        new_script, changes = apply_slurm_binding(
            _SAMPLE_SCRIPT, cpus_per_task=16
        )
        self.assertIn("--cpus-per-task=16", new_script)
        self.assertNotIn("--cpus-per-task=8", new_script)

    def test_exclusive_flag(self):
        new_script, changes = apply_slurm_binding(
            _SAMPLE_SCRIPT, exclusive=True
        )
        self.assertIn("#SBATCH --exclusive", new_script)

    def test_multiple_changes(self):
        new_script, changes = apply_slurm_binding(
            _SAMPLE_SCRIPT,
            ntasks_per_node=4,
            cpus_per_task=32,
            mem_bind="local",
            cpu_bind="cores",
        )
        self.assertIn("--ntasks-per-node=4", new_script)
        self.assertIn("--cpus-per-task=32", new_script)
        self.assertIn("--mem-bind=local", new_script)
        self.assertIn("--cpu-bind=cores", new_script)
        self.assertEqual(len(changes), 4)

    def test_no_changes_when_all_none(self):
        new_script, changes = apply_slurm_binding(_SAMPLE_SCRIPT)
        self.assertEqual(new_script, _SAMPLE_SCRIPT)
        self.assertEqual(changes, [])

    def test_shebang_preserved(self):
        new_script, _ = apply_slurm_binding(_SAMPLE_SCRIPT, mem_bind="local")
        self.assertTrue(new_script.startswith("#!/bin/bash"))

    def test_srun_line_preserved(self):
        new_script, _ = apply_slurm_binding(_SAMPLE_SCRIPT, mem_bind="local")
        self.assertIn("srun ./my_app", new_script)

    def test_insert_when_no_sbatch_lines(self):
        bare_script = "#!/bin/bash\nsrun ./app\n"
        new_script, changes = apply_slurm_binding(bare_script, mem_bind="local")
        self.assertIn("--mem-bind=local", new_script)
        self.assertTrue(any("added" in c for c in changes))


if __name__ == "__main__":
    unittest.main()

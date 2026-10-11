"""Exercise the dispatch/output-directory contract without scientific weights."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import round9_dispatch as dispatch


class TestDispatchContract(unittest.TestCase):
    def test_smoke_and_execute_leave_output_creation_to_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "configs").mkdir(parents=True)
            checkpoints = root / "empty_test_checkpoint_root"
            checkpoints.mkdir()
            (source / dispatch.RUNNER_NAME).write_text(
                "import argparse,json,os\nfrom pathlib import Path\n"
                "p=argparse.ArgumentParser()\n"
                "p.add_argument('--config');p.add_argument('--checkpoint-root')\n"
                "p.add_argument('--run-dir');p.add_argument('--smoke',action='store_true')\n"
                "a=p.parse_args();out=Path(a.run_dir)\n"
                "out.mkdir(parents=True,exist_ok=False)\n"
                "(out/'software_stub.json').write_text(json.dumps({'cpu_only':os.environ.get('CUDA_VISIBLE_DEVICES')==''}))\n",
                encoding="utf-8")
            config = source / dispatch.DEFAULT_CONFIG
            config.write_text(json.dumps({"max_wall_seconds": 10}))
            (source / dispatch.EXPECTED_MANIFEST).write_text("{}")
            smoke = root / "smoke"
            args = argparse.Namespace(source_dir=source, run_dir=smoke,
                                      checkpoint_root=checkpoints, config=dispatch.DEFAULT_CONFIG)
            self.assertEqual(dispatch.action_smoke(args), 0)
            self.assertEqual(json.loads((smoke / "smoke_receipt.json").read_text())["smoke_status"], "COMPLETE")
            run = root / "execute"
            run.mkdir()
            args.run_dir = run
            hashes, digest = dispatch.source_snapshot(source.resolve(), config.resolve())
            dispatch.write_json(run / "launch_receipt.json", {
                "source_dir": str(source.resolve()), "run_dir": str(run.resolve()),
                "checkpoint_root": str(checkpoints.resolve()), "config": dispatch.DEFAULT_CONFIG,
                "source_sha256": hashes, "config_sha256": digest, "max_wall_seconds": 10.0})
            self.assertEqual(dispatch.action_execute(args), 0)
            status = json.loads((run / "job_status.json").read_text())
            self.assertEqual(status["execution_status"], "COMPLETE")
            self.assertTrue(json.loads((run / "certification/software_stub.json").read_text())["cpu_only"])
            with self.assertRaises(FileExistsError):
                dispatch.action_execute(args)


if __name__ == "__main__":
    unittest.main()

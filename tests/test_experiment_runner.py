"""Exercise workflow ordering, CLI compatibility, resume guards, and logging without a GPU."""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from bgc_aggregation.__main__ import build_parser as cli_parser
from script import run_length_extrapolation as runner


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache_path = self.root / 'data/embeddings/length1500_no_mmseqs'
        self.cache_path.mkdir(parents=True)
        self.cache = {'complete': True, 'cache_id': 'test-cache',
                      'encoder': {'embedding_dim': 1152},
                      'experiment': {'protocol': 'length-extrapolation', 'length_cutoff': 1500}}
        (self.cache_path / 'cache.json').write_text(json.dumps(self.cache))
        root_patch = patch.object(runner, 'PROJECT_ROOT', self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.stdout = io.StringIO()
        output_patch = redirect_stdout(self.stdout)
        output_patch.__enter__()
        self.addCleanup(output_patch.__exit__, None, None, None)

    def args(self, *extra):
        return runner.build_parser().parse_args(['--no-plots', *extra])

    def create_run(self, architecture='transformer', seed=42):
        directory = self.root / 'runs' / f'length1500_no_mmseqs_{architecture}_s{seed}'
        directory.mkdir(parents=True)
        metadata = {'cache_id': self.cache['cache_id'], 'experiment': self.cache['experiment'],
                    'initial_checkpoint': None, 'training': {**runner.TRAINING, 'seed': seed},
                    'model': {'embedding_dim': 1152, 'architecture': architecture,
                              'hidden_dim': 128, 'num_layers': 2, 'num_heads': 4,
                              'feedforward_dim': 512, 'dropout': 0.1, 'use_positions': True}}
        (directory / 'run.json').write_text(json.dumps(metadata))
        for name in ('best.pt', 'last.pt'):
            (directory / name).touch()
        return directory

    def test_six_trainings_precede_test_evaluation_and_commands_parse(self):
        with patch.object(runner, 'run_logged') as logged:
            runner.run(self.args())
        commands = [call.args[0] for call in logged.call_args_list]
        self.assertEqual([cmd[4] for cmd in commands], ['train'] * 6 + ['evaluate'] * 6)
        parsed = [cli_parser().parse_args(cmd[4:]) for cmd in commands]
        self.assertEqual([(a.architecture, a.seed) for a in parsed[:6]],
                         [('transformer', 42), ('transformer', 43), ('transformer', 44),
                          ('mlp', 42), ('mlp', 43), ('mlp', 44)])
        for cmd, args in zip(commands, parsed):
            self.assertEqual(cmd[:4], [sys.executable, '-u', '-m', 'bgc_aggregation'])
            self.assertEqual(args.cache, self.cache_path)
            self.assertEqual((args.scope, args.sources), ('target', 'native'))
        for training, evaluation in zip(parsed[:6], parsed[6:]):
            self.assertEqual(training.views, 'fixed')
            self.assertEqual(evaluation.checkpoint, training.output / 'best.pt')
            self.assertEqual(evaluation.split, 'test')
            self.assertEqual(evaluation.bootstrap_replicates, 2000)
            self.assertFalse(evaluation.plots)

    def test_failed_training_prevents_remaining_training_and_evaluation(self):
        error = subprocess.CalledProcessError(7, ['train'])
        with patch.object(runner, 'run_logged', side_effect=[None, error]) as logged:
            with self.assertRaises(subprocess.CalledProcessError):
                runner.run(self.args())
        self.assertEqual(logged.call_count, 2)
        self.assertTrue(all(call.args[0][4] == 'train' for call in logged.call_args_list))

    def test_matching_run_resumes_and_train_stage_never_evaluates(self):
        directory = self.create_run()
        with patch.object(runner, 'run_logged') as logged:
            runner.run(self.args('--stage', 'train'))
        commands = [call.args[0] for call in logged.call_args_list]
        parsed = [cli_parser().parse_args(cmd[4:]) for cmd in commands]
        self.assertEqual(len(parsed), 6)
        self.assertEqual(parsed[0].resume, directory / 'last.pt')
        self.assertTrue(all(args.resume is None for args in parsed[1:]))
        self.assertTrue(all(args.command == 'train' for args in parsed))

    def test_mismatched_last_destination_stops_before_starting_any_training(self):
        directory = self.create_run('mlp', 44)
        path = directory / 'run.json'
        metadata = json.loads(path.read_text())
        metadata['model']['hidden_dim'] = 256
        path.write_text(json.dumps(metadata))
        with patch.object(runner, 'run_logged') as logged:
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                runner.run(self.args())
            logged.assert_not_called()

    def test_evaluate_requires_runs_and_uses_all_best_checkpoints(self):
        with patch.object(runner, 'run_logged') as logged:
            with self.assertRaisesRegex(ValueError, 'No matching trained run'):
                runner.run(self.args('--stage', 'evaluate'))
            logged.assert_not_called()
            for architecture in runner.ARCHITECTURES:
                for seed in runner.SEEDS:
                    self.create_run(architecture, seed)
            runner.run(self.args('--stage', 'evaluate'))
        self.assertEqual(logged.call_count, 6)
        self.assertTrue(all(call.args[0][4] == 'evaluate' for call in logged.call_args_list))

    def test_incomplete_run_is_preserved(self):
        directory = self.create_run()
        (directory / 'last.pt').unlink()
        with patch.object(runner, 'run_logged') as logged:
            with self.assertRaisesRegex(ValueError, 'Incomplete run directory'):
                runner.run(self.args())
            logged.assert_not_called()
        self.assertTrue((directory / 'best.pt').exists())

    def test_logs_append_capture_both_streams_and_propagate_exit_code(self):
        log = self.root / 'experiment.log'
        log.write_text('previous run\n')
        command = [sys.executable, '-u', '-c',
                   "import sys; print('stdout marker'); print('stderr marker', file=sys.stderr); sys.exit(7)"]
        with self.assertRaises(subprocess.CalledProcessError) as raised:
            runner.run_logged(command, log)
        self.assertEqual(raised.exception.returncode, 7)
        content = log.read_text()
        self.assertTrue(content.startswith('previous run\n'))
        for marker in ('stdout marker', 'stderr marker'):
            self.assertIn(marker, content)
            self.assertIn(marker, self.stdout.getvalue())


if __name__ == '__main__':
    unittest.main()

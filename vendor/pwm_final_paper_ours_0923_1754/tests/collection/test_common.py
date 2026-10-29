import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest

from collection.common import write_once, rebase_exclusions, relative_path, resolve_release, check_hashes, digest


def competing_writer(path, barrier, queue):
    barrier.wait(timeout=10)
    try:
        write_once(path, {"complete": True, "payload": "x" * 10000})
        queue.put("won")
    except FileExistsError:
        queue.put("lost")


class CommonTests(unittest.TestCase):
    def test_independent_processes_have_one_complete_winner(self):
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "claim.json")
            barrier, queue = context.Barrier(2), context.Queue()
            workers = [context.Process(target=competing_writer, args=(path, barrier, queue)) for _ in range(2)]
            try:
                for worker in workers: worker.start()
                results = sorted(queue.get(timeout=20) for _ in workers)
                for worker in workers:
                    worker.join(timeout=10)
                    self.assertEqual(worker.exitcode, 0)
                self.assertEqual(results, ["lost", "won"])
                self.assertTrue(json.loads(Path(path).read_text())["complete"])
                self.assertEqual(list(Path(tmp).iterdir()), [Path(path)])
            finally:
                for worker in workers:
                    if worker.is_alive(): worker.terminate(); worker.join()
                queue.close()

    def test_bad_json_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): write_once(Path(tmp) / "bad", {"v": float("nan")})
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_no_clobber(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "receipt"
            write_once(p, {"v": 1})
            with self.assertRaises(FileExistsError): write_once(p, {"v": 2})
            self.assertEqual(json.loads(p.read_text()), {"v": 1})

    def test_rebase_preserves_boundary_relative_and_input(self):
        source = {('/a/data/shard.tar', 'k'): {1}, ('/a/database/shard.tar', 'k'): {2},
                  ('shard.tar', 'k'): {3}}
        result = rebase_exclusions(source, '/a/data', '/b/data')
        self.assertEqual(result[('/b/data/shard.tar', 'k')], {1})
        self.assertEqual(result[('/a/database/shard.tar', 'k')], {2})
        self.assertEqual(result[('shard.tar', 'k')], {3})
        result[('/b/data/shard.tar', 'k')].add(9)
        self.assertEqual(source[('/a/data/shard.tar', 'k')], {1})

    def test_whole_batch_overrides_rows_in_both_orders(self):
        for source in ({('/a/x', 'k'): None, ('/b/x', 'k'): {2}},
                       {('/b/x', 'k'): {2}, ('/a/x', 'k'): None}):
            self.assertIsNone(rebase_exclusions(source, '/a', '/b')[('/b/x', 'k')])

    def test_rows_merge(self):
        source = {('/a/x', 'k'): {1}, ('/b/x', 'k'): {2}}
        self.assertEqual(rebase_exclusions(source, '/a', '/b')[('/b/x', 'k')], {1, 2})
        self.assertIsNone(rebase_exclusions(None, '/a', '/b'))

    def test_traversal_absolute_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('../escape', '/etc/passwd', 'x/../../escape'):
                with self.assertRaises(ValueError): relative_path(root, name)
            (root / 'link').symlink_to('/tmp', target_is_directory=True)
            with self.assertRaises(ValueError): relative_path(root, 'link/file')

    def test_locations_do_not_change_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_once(root / 'a/shard', {'row': 1})
            write_once(root / 'b/shard', {'row': 1})
            catalog = {'r1': dict(release_id='r1', domain='pickup', canonical_root=str(root / 'a'), shards=['shard'])}
            a = resolve_release(catalog, 'r1', {'r1': root / 'a'})
            b = resolve_release(catalog, 'r1', {'r1': root / 'b'})
            self.assertEqual(a['release_id'], b['release_id'])
            self.assertEqual(digest(a['shards'][0]), digest(b['shards'][0]))
            with self.assertRaises(ValueError): resolve_release(catalog, 'r1', {'r1': root / 'missing'})

    def test_corrupt_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_once(Path(tmp) / 'x', {'v': 1})
            with self.assertRaises(ValueError): check_hashes(tmp, {'x': '0' * 64})


if __name__ == '__main__': unittest.main()

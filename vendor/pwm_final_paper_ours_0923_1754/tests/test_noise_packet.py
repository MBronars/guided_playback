"""Exercise packet launching with conflicting Python packages in two snapshots."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys


def test_entry_snapshot_controls_module_import_and_qualification(tmp_path):
    old, new = tmp_path / 'old', tmp_path / 'new'
    for source, marker in ((old, 'old'), (new, 'new')):
        package = source / 'guidance'
        package.mkdir(parents=True)
        (package / '__init__.py').write_text('')
        (package / 'probe.py').write_text(
            f"from pathlib import Path\nprint({marker!r}, Path.cwd(), __file__)\n")
    passed, failed = tmp_path / 'passed.json', tmp_path / 'failed.json'
    passed.write_text(json.dumps({'status': 'passed'}))
    failed.write_text(json.dumps({'status': 'failed'}))
    future = (datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat()
    entries = []
    for name, extra in (
        ('default', {}),
        ('override', {'source_override': str(new), 'qualification': str(passed)}),
        ('rejected', {'source_override': str(new), 'qualification': str(failed)}),
    ):
        entries.append(dict(name=name, scene='cpu', arm=name, timeout_seconds=10,
                            command=[sys.executable, '-m', 'guidance.probe'], **extra))
    manifest = tmp_path / 'packet.json'
    manifest.write_text(json.dumps(dict(source_root=str(old), gpu_uuid='',
        qualification=str(passed), deadline_utc=future, stop_launch_utc=future,
        entries=entries)))
    script = Path(__file__).resolve().parents[1] / 'scripts/dev/run_noise_packet.py'
    subprocess.run([sys.executable, str(script), '--manifest', str(manifest)], check=True)
    status = json.loads((tmp_path / 'packet_status.json').read_text())
    assert [e['state'] for e in status['entries']] == ['completed', 'completed', 'failed']
    assert (tmp_path / 'default.log').read_text().split()[0] == 'old'
    assert (tmp_path / 'override.log').read_text().split()[0] == 'new'
    assert str(new) in (tmp_path / 'override.log').read_text()
    assert status['entries'][1]['source_root'] == str(new)
    assert 'Entry qualification failed' in status['entries'][2]['error']
    assert 'pid' not in status['entries'][2]
    assert not (tmp_path / 'rejected.log').exists()
